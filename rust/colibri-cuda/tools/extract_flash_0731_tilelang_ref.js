#!/usr/bin/env node
'use strict';

// Extract the original PTX debug record from the two locked TileLang cache
// entries used by the official Flash-0731 FP4 GEMMs. The Rust runtime only
// consumes the emitted PTX and manifest; it never imports Python or TileLang.
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const RELEASE_SHA = '9e165c30e2704aec5d9d593cce3eebd58bbef1cb';
const ENTRY = 'fp4_gemm_kernel__kernel';
const VARIANTS = [
  {
    name: 'n2048_k4096',
    cacheKey: '26bdc12be1dd8bc0561e4d5663179414244d66aa2583fa3dd64de50a6531dc5d',
    deviceSourceSha256: '4dfbd21891c4173465c42f90d48b16d5a4ac486f67e51bad1040377165c19f20',
    hostSourceSha256: '57cc310f114d21e54ff2c3fe15b32adb2eda70fb3cc1d43ad5d206a0ba7e0d3e',
    executableSha256: '04911b0f360e9d57a832d3019b1dcc5433d62ea46937334b93b9d1d6dfb206c2',
    n: 2048,
    k: 4096,
    scaleColumns: 128,
    sharedBytes: 17360,
  },
  {
    name: 'n4096_k2048',
    cacheKey: 'f6ffd2a12d4b01853284efbffaad7b5a6e3e10afa155aa2dffc86c6ff8e4e8f6',
    deviceSourceSha256: 'b206460df8f94699661ac9f5c87b5965096251541f1370bbacd160442666fbfc',
    hostSourceSha256: 'b51362482801c14e03d7cc4dfc154a57c86c0b45692398d8e8ad6cbfb04bd6f2',
    executableSha256: '817462d29dab84fc3278cc9147964735ab808dd0926dbf6ab6fe070db0d15df8',
    n: 4096,
    k: 2048,
    scaleColumns: 64,
    sharedBytes: 17360,
  },
];

function usage() {
  throw new Error('usage: extract_flash_0731_tilelang_ref.js --model-dir <0731-root> --cache-root <tilelang-cache> --out <empty-or-new-dir>');
}

function args(argv) {
  const parsed = {};
  for (let index = 2; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!['--model-dir', '--cache-root', '--out'].includes(key) || !value) usage();
    parsed[key] = value;
  }
  if (Object.keys(parsed).length !== 3) usage();
  return parsed;
}

function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function readVerified(file, expected) {
  const bytes = fs.readFileSync(file);
  const actual = sha256(bytes);
  if (actual !== expected) throw new Error(`SHA-256 mismatch for ${file}: expected ${expected}, got ${actual}`);
  return bytes;
}

function extractPtx(executable, variant) {
  const begin = Buffer.from('.version 9.0\0.target sm_89\0.address_size 64\0');
  const start = executable.indexOf(begin);
  const vendorNote = executable.indexOf(Buffer.from('NVIDIA Corp\0'), start);
  const endAt = executable.lastIndexOf(Buffer.from('\0}\0'), vendorNote);
  if (start < 0 || endAt < 0) throw new Error(`${variant.name}: PTX NUL/length boundary not found`);
  const raw = executable.subarray(start, endAt + 2);
  const ptx = Buffer.from(raw.toString('utf8').replace(/\0/g, '\n'), 'utf8');
  if (ptx.includes(0) || !ptx.subarray(0, 64).toString('utf8').startsWith('.version 9.0\n.target sm_89\n.address_size 64')) {
    throw new Error(`${variant.name}: invalid PTX header or retained NUL`);
  }
  const text = ptx.toString('utf8');
  const parameters = [0, 1, 2, 3, 4].map((index) => `.param .u64 ${ENTRY}_param_${index}`);
  const required = [
    `.visible .entry ${ENTRY}`,
    ...parameters,
    `.param .u32 ${ENTRY}_param_5`,
    '.maxntid 128, 1, 1',
  ];
  for (const token of required) if (!text.includes(token)) throw new Error(`${variant.name}: missing ${token}`);
  if (text.indexOf('.visible .entry') !== text.lastIndexOf('.visible .entry')) throw new Error(`${variant.name}: expected one PTX entry`);
  return { ptx, sourceOffset: start, sourceLength: raw.length };
}

function main() {
  const parsed = args(process.argv);
  const modelDir = path.resolve(parsed['--model-dir']);
  const cacheRoot = path.resolve(parsed['--cache-root']);
  const out = path.resolve(parsed['--out']);
  const kernelPath = path.join(modelDir, 'inference', 'kernel.py');
  if (!kernelPath.includes(RELEASE_SHA)) throw new Error(`model path does not carry locked release ${RELEASE_SHA}`);
  const kernelSha256 = sha256(fs.readFileSync(kernelPath));
  if (fs.existsSync(out) && fs.readdirSync(out).length !== 0) throw new Error(`output directory must be empty: ${out}`);
  fs.mkdirSync(out, { recursive: true });

  const entries = {};
  for (const variant of VARIANTS) {
    const directory = path.join(cacheRoot, variant.cacheKey);
    const device = readVerified(path.join(directory, 'device_kernel.cu'), variant.deviceSourceSha256);
    const deviceText = device.toString('utf8');
    const hostText = readVerified(path.join(directory, 'host_kernel.cu'), variant.hostSourceSha256).toString('utf8');
    const shapeNeedles = [
      `fp4_gemm_kernel__B_shape)[0]) != ${variant.n}`,
      `fp4_gemm_kernel__B_shape)[1]) * 2) != ${variant.k}`,
      `fp4_gemm_kernel__scales_b_shape)[1]) != ${variant.scaleColumns}`,
      `__launch_bounds__(128, 1) ${ENTRY}`,
    ];
    for (const needle of shapeNeedles) {
      const source = needle.includes('_shape)') ? hostText : deviceText;
      if (!source.includes(needle)) throw new Error(`${variant.name}: source ABI mismatch: ${needle}`);
    }
    const executable = readVerified(path.join(directory, 'executable.so'), variant.executableSha256);
    const extracted = extractPtx(executable, variant);
    const filename = `${variant.name}.sm89.ptx`;
    fs.writeFileSync(path.join(out, filename), extracted.ptx, { flag: 'wx' });
    entries[variant.name] = {
      cache_key: variant.cacheKey,
      executable_sha256: variant.executableSha256,
      device_source_sha256: variant.deviceSourceSha256,
      host_source_sha256: variant.hostSourceSha256,
      ptx_file: filename,
      ptx_sha256: sha256(extracted.ptx),
      ptx_source_offset: extracted.sourceOffset,
      ptx_source_length: extracted.sourceLength,
      n: variant.n,
      k: variant.k,
      b_packed_shape: [variant.n, variant.k / 2],
      scales_b_shape: [variant.n, variant.scaleColumns],
      grid: { x: `ceil(${variant.n}/128)`, y: 'ceil(M/32)', z: 1 },
      block: [128, 1, 1],
      dynamic_shared_bytes: variant.sharedBytes,
    };
  }
  const manifest = {
    schema: 'colibri-flash-tilelang-ref-v1',
    artifact_kind: 'official_cached_ptx',
    claim_boundary: 'PTX is extracted from the locked official TileLang cache. It is not a separately rebuilt cubin and is diagnostic-only until a parity gate passes.',
    release_sha: RELEASE_SHA,
    kernel_py_sha256: kernelSha256,
    target: 'sm_89',
    ptx_version: '9.0',
    entry: ENTRY,
    raw_argument_order: [
      'A: ptr<fp8_e4m3fn>',
      'B: ptr<fp4_e2m1fn_x2 packed along K>',
      'C: ptr<bf16>',
      'scales_a: ptr<fp8_e8m0>',
      'scales_b: ptr<fp8_e8m0>',
      'M: u32',
    ],
    entries,
  };
  fs.writeFileSync(path.join(out, 'tilelang-ref-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, { flag: 'wx' });
  console.log(JSON.stringify({ status: 'PASS', manifest: path.join(out, 'tilelang-ref-manifest.json'), entries: Object.keys(entries) }));
}

main();
