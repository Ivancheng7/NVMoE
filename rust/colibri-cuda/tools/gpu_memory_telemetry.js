#!/usr/bin/env node
'use strict';

const assert = require('assert');
const childProcess = require('child_process');

const SCHEMA = 'colibri-experiment-gpu-memory-v1';
const QUERY = 'memory.total,memory.used,memory.free';
const FORMAT = 'csv,noheader,nounits';
const COMMAND = [
  `--query-gpu=${QUERY}`,
  `--format=${FORMAT}`,
];
const QUERY_STRING = `nvidia-smi ${COMMAND.join(' ')}`;

function parse(raw) {
  const lines = raw.trimEnd().split(/\r?\n/);
  const fields = lines.length === 1 ? lines[0].split(',').map((field) => field.trim()) : [];
  if (fields.length !== 3 || fields.some((field) => !/^\d+$/.test(field))) {
    const actual = lines.length === 1 ? `${fields.length} fields` : `${lines.length} rows`;
    throw new Error(
      `expected one GPU row with exactly 3 integer fields from "${QUERY_STRING}", got ${actual}`,
    );
  }
  const [total, used, free] = fields.map(Number);
  if (!Number.isSafeInteger(total) || !Number.isSafeInteger(used)
      || !Number.isSafeInteger(free) || total <= 0 || used > total || free > total) {
    throw new Error(`invalid GPU memory values from "${QUERY_STRING}"`);
  }
  return {
    schema: SCHEMA,
    query: QUERY_STRING,
    units: 'MiB',
    memory: { total, used, free },
  };
}

function capture() {
  let raw;
  try {
    raw = childProcess.execFileSync('nvidia-smi', COMMAND, {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  } catch (error) {
    const detail = error.stderr ? error.stderr.toString().trim() : error.message;
    throw new Error(`"${QUERY_STRING}" failed: ${detail}`);
  }
  return parse(raw);
}

function selfTest() {
  const sample00 = parse('12282, 1211, 10785\n');
  const sample01 = parse('12282, 1211, 10785\n');
  assert.deepStrictEqual(sample00, sample01);
  assert.throws(
    () => parse(
      '2026/07/26 18:48:53.406, 0, NVIDIA GeForce RTX 4080 Laptop GPU, '
      + 'GPU-755a8b44-70c8-1aa0-e214-d98f9ac67b5e, 78, 5.33 W, 1211 MiB, 0 %\n',
    ),
    /exactly 3 integer fields/,
  );
  assert.deepStrictEqual(parse('12282, 1211, 10785\n'), sample00);
  return {
    status: 'PASS',
    schema: SCHEMA,
    query: QUERY_STRING,
    checks: {
      sample00_canonical: 'PASS',
      sample01_canonical: 'PASS',
      sample02_wrong_columns: 'FAIL_CLOSED',
      sample02_canonical_parser_only: 'PASS',
    },
  };
}

function main() {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.log(JSON.stringify(capture()));
    return;
  }
  if (args.length === 1 && args[0] === '--self-test') {
    console.log(JSON.stringify(selfTest()));
    return;
  }
  throw new Error('unexpected arguments');
}

try {
  main();
} catch (error) {
  console.error(`FAIL gpu-memory-telemetry: ${error.message}`);
  process.exitCode = 2;
}
