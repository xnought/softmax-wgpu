import { GPU } from "./webgpu-compute.js";

dev();

async function dev() {
	const gpu = await GPU.init();
	gpu.printDeviceInfo();

	const shape = [1000, 1000];
	const a = genRangeData(shape);
	const aDev = gpu.memAlloc(a.byteLength);
	gpu.memcpyHostToDevice(aDev, a);
	console.log("INPUT");
	await printMatrix(gpu, aDev, shape);

	const result = gpu.memAlloc(a.byteLength);
	onlineSafeSoftmax(gpu, result, aDev, shape);
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
		cpuData[i] = 1;
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

/**
 * First max, then sub, exp, then sum, then divide.
 * @param {GPU} gpu
 * @param {GPUBuffer} dst
 * @param {GPUBuffer} src
 * @param {[number, number]} shape
 */
function safeSoftmax2D(gpu, dst, src, shape) {
	const maxLength = shape[0] * shape[1];
	const main = gpu
		.SourceModule(
			/*wgsl*/ `
		@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
		@group(0) @binding(1) var<storage, read> src: array<f32>;

		@compute @workgroup_size(16, 16)
		fn main(@builtin(global_invocation_id) gid : vec3u) {
			let i = gid.x;
			let j = gid.y;
			if(i < ${shape[0]} && j < ${shape[1]}) {
				let ij = i*${shape[1]} + j;
				var max: f32 = src[i*${shape[1]}];
				for(var k: u32 = 0; k < ${shape[1]}; k++) {
					let ik = i*${shape[1]} + k;
					let curMax = src[ik];
					if(curMax > max) {
						max = curMax;
					}
				}
				var denom: f32 = 0.0;
				for(var k: u32 = 0; k < ${shape[1]}; k++) {
					let ik = i*${shape[1]} + k;
					denom += exp(src[ik] - max);
				}
				dst[ij] = exp(src[ij] - max) / denom;
			}
		}
	`
		)
		.getFunction("main");

	main([Math.ceil(shape[0] / 16), Math.ceil(shape[1] / 16)], dst, src);
}

/**
 * max and sum without multiples memory accesses
 * @param {GPU} gpu
 * @param {GPUBuffer} dst
 * @param {GPUBuffer} src
 * @param {[number, number]} shape
 */
function onlineSafeSoftmax(gpu, dst, src, shape) {
	const maxLength = shape[0] * shape[1];
	const main = gpu
		.SourceModule(
			/*wgsl*/ `
		@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
		@group(0) @binding(1) var<storage, read> src: array<f32>;

		@compute @workgroup_size(16, 16)
		fn main(@builtin(global_invocation_id) gid : vec3u) {
			let i = gid.x;
			let j = gid.y;
			if(i < ${shape[0]} && j < ${shape[1]}) {
				var _max: f32 = src[i*${shape[1]}];
				var denom: f32 = 0.0;
				// Can I move this to be shared across threads?
				for(var k: u32 = 0; k < ${shape[1]}; k++) {
					let ik = i*${shape[1]} + k;
					let val = src[ik];
					if(val > _max) {
						denom *= exp(_max - val);
						_max = val;
					} 
					denom += exp(val-_max);
				}
				let ij = i*${shape[1]} + j;
				dst[ij] = exp(src[ij] - _max) / denom;
			}
		}
	`
		)
		.getFunction("main");

	main([Math.ceil(shape[0] / 16), Math.ceil(shape[1] / 16)], dst, src);
}
