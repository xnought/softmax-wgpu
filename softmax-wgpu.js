import { GPU } from "./webgpu-compute.js";

dev();

async function dev() {
	const gpu = await GPU.init();
	gpu.printDeviceInfo();

	const shape = [5, 4];
	const a = genRangeData(shape);
	const aDev = gpu.memAlloc(a.byteLength);
	gpu.memcpyHostToDevice(aDev, a);
	console.log("INPUT");
	await printMatrix(gpu, aDev, shape);

	const result = gpu.memAlloc(a.byteLength);
	safeSoftmax(gpu, result, aDev, shape);
	await gpu.deviceSynchronize();
	await printMatrix(gpu, result, shape);
}

async function printMatrix(gpu, buffer, shape) {
	const cpuBuffer = await gpu.mapGPUToCPU(buffer, Float32Array);
	for (let i = 0; i < shape[0]; i++) {
		let inner = "";
		for (let j = 0; j < shape[1]; j++) {
			inner += `${cpuBuffer[i * shape[1] + j].toFixed(3)} `;
		}
		console.log(inner);
	}
	console.log("\n");
}

function genRangeData([rows, cols]) {
	const cpuData = new Float32Array(rows * cols);
	for (let i = 0; i < cpuData.length; i++) {
		cpuData[i] = Math.random();
	}
	return cpuData;
}

/**
 * First exp, then sum, then divide.
 * @param {GPU} gpu
 * @param {GPUBuffer} dst
 * @param {GPUBuffer} src
 * @param {[number, number]} shape
 */
function naiveSoftmax(gpu, dst, src, shape) {
	const maxLength = shape[0] * shape[1];
	const main = gpu
		.SourceModule(
			/*wgsl*/ `
		@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
		@group(0) @binding(1) var<storage, read> src: array<f32>;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid : vec3u) {
			let i = gid.x;
			if(i < ${shape[0]}) {
				var denom: f32 = 0.0;
				for(var j: u32 = 0; j < ${shape[1]}; j++) {
					let ij = i*${shape[1]} + j;
					denom += exp(src[ij]);
				}
				for(var j: u32 = 0; j < ${shape[1]}; j++) {
					let ij = i*${shape[1]} + j;
					dst[ij] = exp(src[ij]) / denom;
				}
			}
		}
	`
		)
		.getFunction("main");

	main([Math.ceil(maxLength / 256)], dst, src);
}

/**
 * First max, then sub, exp, then sum, then divide.
 * @param {GPU} gpu
 * @param {GPUBuffer} dst
 * @param {GPUBuffer} src
 * @param {[number, number]} shape
 */
function safeSoftmax(gpu, dst, src, shape) {
	const maxLength = shape[0] * shape[1];
	const main = gpu
		.SourceModule(
			/*wgsl*/ `
		@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
		@group(0) @binding(1) var<storage, read> src: array<f32>;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid : vec3u) {
			let i = gid.x;
			if(i < ${shape[0]}) {
				var max: f32 = src[i*${shape[1]}];
				for(var j: u32 = 0; j < ${shape[1]}; j++) {
					let ij = i*${shape[1]} + j;
					let curMax = src[ij];
					if(curMax > max) {
						max = curMax;
					}
				}
				var denom: f32 = 0.0;
				for(var j: u32 = 0; j < ${shape[1]}; j++) {
					let ij = i*${shape[1]} + j;
					denom += exp(src[ij] - max);
				}
				for(var j: u32 = 0; j < ${shape[1]}; j++) {
					let ij = i*${shape[1]} + j;
					dst[ij] = exp(src[ij] - max) / denom;
				}
			}
		}
	`
		)
		.getFunction("main");

	main([Math.ceil(maxLength / 256)], dst, src);
}
