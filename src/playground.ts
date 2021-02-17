/* Copyright 2016 Google Inc. All Rights Reserved.
Modifications Copyright 2021 Modupalli Yasodhara.

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
/*----Test script code Begins---*/
export class EntryPoint {
    constructor() {
        console.log("constructor is called!")
    }

    showalert() {
        alert("Success!!");
    }
};

(window as any).showalert = function () {
    let hero = new EntryPoint();
    hero.showalert();
};
/*----Test script code ends ---*/

import './assets/styles.css';
import * as d3 from 'd3';
import * as seedrandom from 'seedrandom';
import {
    State,
    datasets,
    regDatasets,
    problems,
    getKeyFromValue,
    Problem
  } from './state';
  import { DataGenerator, Example2D, shuffle, isValid } from './dataset';


  const NUM_SAMPLES_CLASSIFY = 400;
  const NUM_SAMPLES_REGRESS = 800;
  // Size of the heatmaps.
  const SIDE_LENGTH = 300;
  // # of points per direction.
  const DENSITY = 50;
  const NUM_VISIBLE_TREES = 16;
  
  const state = State.deserializeState();
  const xDomain: [number, number] = [-6, 6];
  // Label values must be scaled before and after training since RF impl does not
  // accepts negative values.
  const inputScale = d3
    .scaleLinear()
    .domain([-1, 1])
    .range([0, 1]);
  const outputScale = d3
    .scaleLinear()
    .domain([0, 1])
    .range([-1, 1]);

    const colorScale = d3
    .scaleLinear<string, number>()
    .domain([-1, 0, 1])
    .range(['#f59322', '#e8eaeb', '#0877bd'])
    .clamp(true);
 
  let trainWorker: Worker;
  let options: any;
  let Method: any;
  let data: Example2D[];
  let uploadedData: Example2D[];
  let trainData: Example2D[];
  let testData: Example2D[];
  let metricList = [];
  let getMetrics: (yPred: number[], yTrue: number[]) => any;
  let trainMetrics;
  let testMetrics;
  let mainBoundary: number[][];
  let treeBoundaries: number[][][];


/**
 * Shows busy indicators in the UI as something is running in the background.
 * They include making all heatmaps opaque and showing a progress indicator next
 * to the cursor.
 * @param {boolean} loading True if something is running in the background
 */
function isLoading(loading: boolean) {
    d3.select('#main-heatmap canvas')
      .style('opacity', loading ? 0.2 : 1);
    d3.select('#main-heatmap svg')
      .style('opacity', loading ? 0.2 : 1);
    d3.selectAll('.tree-heatmaps-container canvas')
      .style('opacity', loading ? 0.2 : 1);
    d3.selectAll('*')
      .style('cursor', loading ? 'progress' : 'null');
}

function isClassification() {
    return state.problem === Problem.CLASSIFICATION;
}
