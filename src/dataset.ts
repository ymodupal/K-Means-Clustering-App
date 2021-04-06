/* Copyright 2016 Google Inc. All Rights Reserved.
Modifications Copyright 2021 Yasodhara Modupalli.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as d3 from "d3";
import { Schema, Validator } from 'jsonschema';
import * as tf from '@tensorflow/tfjs';

/**
 * A two dimensional example: x and y coordinates with the label.
 */
export type Example2D = {
  x: number;
  y: number;
  label: number;
  cluster: number;
  IsCentroid: boolean;
};

export type Point = {
  x: number;
  y: number;
};

const schema: Schema = {
  type: 'array',
  maxItems: 1000,
  items: {
    type: 'object',
    required: ['x', 'y', 'label'],
    properties: {
      x: {
        type: 'number',
        minimum: -6,
        maximum: 6
      },
      y: {
        type: 'number',
        minimum: -6,
        maximum: 6
      },
      label: {
        enum: [-1, 1]
      }
    }
  }
};

/**
 * Check if the JSON data is in valid format based on the predefined schema.
 * @param data input data.
 * @returns {boolean} true if the data is valid.
 */
export function isValid(data: any): boolean {
  return new Validator().validate(data, schema).valid;
}

/**
 * Shuffles the array using Fisher-Yates algorithm. Uses the seedrandom
 * library as the random generator.
 */
export function shuffle(array: any[]): void {
  let counter = array.length;
  let temp = 0;
  let index = 0;

  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = Math.floor(Math.random() * counter);
    // Decrease counter by 1
    counter--;
    // And swap the last element with it
    temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

export type DataGenerator = (numSamples: number, noise: number) => Example2D[];

export function classifyTwoGaussData(numSamples: number, noise: number):
Example2D[] {
  const points: Example2D[] = [];

  const varianceScale = d3
    .scaleLinear()
    .domain([0, 0.5])
    .range([0.5, 4]);
  const variance = varianceScale(noise);

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      const x = normalRandom(cx, variance);
      const y = normalRandom(cy, variance);
      const cluster: number = 0;
      points.push(Get2dPoint(x, y, label, cluster, false));
    }
  }
  genGauss(0, 2, 1); // Gaussian with positive examples.
  genGauss(-3, -2, -1); // Gaussian with negative examples.
  genGauss(4, -2, -1);
  return points;
}

export function classifySpiralData(numSamples: number, noise: number):
Example2D[] {
  const points: Example2D[] = [];
  const n = numSamples / 2;

  function genSpiral(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      const r = i / n * 5;
      const t = 1.75 * i / n * 2 * Math.PI + deltaT;
      const x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      const y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      const cluster: number = 0;
      points.push(Get2dPoint(x, y, label, cluster, false));
    }
  }

  genSpiral(0, 1); // Positive examples.
  genSpiral(Math.PI, -1); // Negative examples.
  return points;
}

export function classifyCircleData(numSamples: number, noise: number):
Example2D[] {
  const points: Example2D[] = [];
  const radius = 5;
  function getCircleLabel(p: Point, center: Point) {
    return (dist(p, center) < (radius * 0.5)) ? 1 : -1;
  }

  function genCircle(mean: number, variance: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      const r = randUniform(mean, variance);
      const angle = randUniform(0, 2 * Math.PI);
      const x = r * Math.cos(angle);
      const y = r * Math.sin(angle);
      const label = getCircleLabel(
        { x: x , y: y },
        { x: 0, y: 0 },
      );
      const cluster: number = 0;
      points.push(Get2dPoint( x, y, label, cluster, false));
    }
  }

  genCircle(0, radius * (0.5 + noise));
  genCircle(radius * (0.7 + noise), radius);
  return points;
}

export function classifyMoonData(numSamples: number, noise: number):
Example2D[] {
  const points: Example2D[] = [];
  const radius = 3.8;
  function getMoonLabel(p: Point, center: Point) {
    return (dist(p, center) < (radius * 0.5)) ? 1 : -1;
  }
  const samples_out = Math.floor(numSamples/2); //integer division
  const samples_in = numSamples - samples_out;

  function genUpperMoon(mean: number, variance: number) {
    for (let i = 0; i < samples_out; i++) {
      const r = randUniform(mean, variance); // * 0.x make it smaller in width
      const angle = randUniform(1, Math.PI + 1.9); 
      const x =  r * Math.sin(angle) - 1.2;  // + values move the moon to right
      const y =  r * Math.cos(angle) - 0.5; // negative values move the half circle down
      const label = getMoonLabel(
        { x: x, y: y },
        { x: 0, y: 0 },
      );
      const cluster: number = 0;
      points.push(Get2dPoint(x, y, label, cluster, false));
    }
  }
  function genLowerMoon(mean: number, variance: number) {
    for (let i = 0; i < samples_in; i++) {
      const r = randUniform(mean, variance);
      const angle = randUniform(1.1, Math.PI + 2.1);
      const x =  r * Math.sin(angle) + 1.5;
      const y =  r * Math.cos(angle) - 1.5;
      const label = getMoonLabel(
        { x: x, y: y },
        { x: 0, y: 0 },
      );
      const cluster: number = 0;
      points.push(Get2dPoint(x, y, label, cluster, false));
    }
  }
  genUpperMoon(-radius, -radius * (0.8 - noise));
  genLowerMoon(radius, radius * (0.8 - noise));
  return points;
}

export function classifyAnisoData(numSamples: number, noise: number):
Example2D[] {
  const points: Example2D[] = [];

  const varianceScale = d3
    .scaleLinear()
    .domain([0, 0.5])
    .range([0.5, 4]);
  const variance = varianceScale(noise);
  //const transform = [[0.6, -0.6], [-0.6, 0.9]];

  function genAniso(cx: number, cy: number, label: number, transform: number[][]) {
    let matA = [];
    for (let i = 0; i < numSamples / 2; i++) {
      const x = Math.sin(Math.PI) + normalRandom(cx, variance);
      const y = Math.cos(Math.PI) + normalRandom(cy, variance);
      matA.push([x, y]);
    }

    let mulRes = <number[][]>tf.matMul(matA, transform).arraySync();
    //console.log(mulRes);
    const cluster: number = 0;
    mulRes.forEach(item => {
      const x: number = item[0];
      const y: number = item[1];
      points.push(Get2dPoint(x, y, label, cluster, false));
    });
  }
  genAniso(0, 4, 1, [[0.7, -0.7], [-0.6, 1.0]]);
  genAniso(-6, -2, -3, [[0.7, -0.7], [-0.6, 1.0]]);
  genAniso(-1, -3, -4, [[0.7, -0.7], [-0.6, 1.0]]);
  return points;
}

export function classifyXORData(numSamples: number, noise: number):
Example2D[] {
  const points: Example2D[] = [];
  const padding = 0.35;
  const radius = 5;
  function getXORLabel(p: Point) {
    return p.x * p.y >= 0 ? 1 : -1;
  }

  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    x += x > 0 ? padding : -padding;

    let y = randUniform(-radius, radius);
    y += y > 0 ? padding : -padding;

    const noiseX = randUniform(-radius, radius) * noise;
    const noiseY = randUniform(-radius, radius) * noise;
    const label = getXORLabel({ x: x + noiseX, y: y + noiseY });
    const cluster: number = 0;
    
    points.push(Get2dPoint(x, y, label, cluster, false));
  }
  return points;
}

/**
 * Returns a sample from a uniform [a, b] distribution.
 * Uses the seedrandom library as the random generator.
 */
function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}

/**
 * Samples from a normal distribution. Uses the seedrandom library as the
 * random generator.
 *
 * @param mean The mean. Default is 0.
 * @param variance The variance. Default is 1.
 */
function normalRandom(mean = 0, variance = 1): number {
  let v1: number;
  let v2: number;
  let s: number;

  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 ** 2 + v2 ** 2;
  } while (s > 1);

  const result = Math.sqrt(-2 * Math.log(s) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}

/** Returns the eucledian distance between two points in space. */
function dist(a: Point, b: Point): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx ** 2 + dy ** 2);
}

export function Get2dPoint(x: number, y: number, label: number, cluster: number, IsCentroid: boolean): Example2D  {
  var obj: Example2D = {
    x: x,
    y: y,
    label: label,
    cluster: cluster,
    IsCentroid: IsCentroid
  } 
  return obj;
}