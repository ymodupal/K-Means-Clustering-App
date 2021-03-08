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

import './assets/styles.css';
import * as d3 from 'd3';
import * as seedrandom from 'seedrandom';
import { HeatMap } from './heatmap';
import {
    State,
    datasets,
    regDatasets,
    problems,
    getKeyFromValue,
    Problem
  } from './state';
import { DataGenerator, Example2D, shuffle, isValid } from './dataset';
import * as utils from './utils';
import kmeans from 'ml-kmeans';

const NUM_SAMPLES_CLASSIFY = 400;
const NUM_SAMPLES_REGRESS = 800;
// Size of the heatmaps.
const SIDE_LENGTH = 300;
// # of points per direction.
const DENSITY = 50;
const NUM_VISIBLE_TREES = 16;
  
const state = State.deserializeState();
const xDomain: [number, number] = [-6, 6];

const colorsPreset = ['#e8eaeb', '#0877bd', '#c27ba0', '#8e7cc3', '#6d9eeb', '#76a5af', '#93c47d', '#f6b26b'];
// Label values must be scaled before and after training since RF impl does not
// accepts negative values.

const colorScale = d3
    .scaleLinear<string, number>()
    .domain([-1, 0, 1])
    //.range(['#f59322', '#e8eaeb', '#0877bd'])
    .range(['#808B96', '#808B96', '#808B96'])
    .clamp(true);

// Plot the main heatmap.
const mainHeatMap = new HeatMap(
    SIDE_LENGTH,
    DENSITY,
    xDomain,
    xDomain,
    d3.select('#main-heatmap'),
    { showAxes: true }
);

// Plot the main heatmap.
const outputHeatMap = new HeatMap(
  SIDE_LENGTH,
  DENSITY,
  xDomain,
  xDomain,
  d3.select('#output-heatmap'),
  { showAxes: true }
);

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


/**
 * Prepares the UI on startup.
 */
function makeGUI() {
  d3.select('#start-button').on('click', () => {
    isLoading(true);
    
    let centroidIndexes = utils.randArray(0, trainData.length-1, state.clusters);

    let centroidArray = [];
    centroidIndexes.forEach(i => {
      centroidArray.push(trainData[i]);
    });

    let inputData = get2dArray(trainData);
    let noOfClusters = parseInt(state.clusters.toString());
    let seedForKmeans = utils.getRandomInt(0,trainData.length-1);
    console.log('Clusters = '+noOfClusters + ', Seed = ' + seedForKmeans);

    //let centers = d3.select('#clusterCount').value();
    let ans = kmeans(inputData, noOfClusters, 
      { seed: seedForKmeans});
    console.log(ans);
    
    setClusterIndexes(ans.clusters);
    
    outputHeatMap.setColorScale();
    outputHeatMap.updatePoints(trainData);

    //deleted train worker logic.
    isLoading(false);
  });

  let centers = d3.select("#clusterCount").on("change", function() {
   state.clusters = (this as any).value;
  // state.serialize();
  });

  /* Data column */
  d3.select('#data-regen-button').on('click', () => {
    generateData();
    reset();
  });

  const dataThumbnails = d3.selectAll('canvas[data-dataset]');
  dataThumbnails.on('click', function () {
    const newDataset = datasets[(this as HTMLElement).dataset.dataset!];
    if (newDataset === state.dataset) {
      return; // No-op.
    }
    state.dataset = newDataset;
    dataThumbnails.classed('selected', false);
    d3.select(this).classed('selected', true);
    generateData();
    reset();
  });

  const datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed('selected', true);

  const regDataThumbnails = d3.selectAll('canvas[data-regDataset]');
  regDataThumbnails.on('click', function () {
    const newDataset = regDatasets[(this as HTMLCanvasElement).dataset.regdataset!];
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset = newDataset;
    regDataThumbnails.classed('selected', false);
    d3.select(this).classed('selected', true);
    generateData();
    reset();
  });

  const regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed('selected', true);

  d3.select('#file-input')
    .on('input', async function() {
      const element = this as HTMLInputElement;
      const file = element.files![0];

      if (file.type !== 'application/json') {
        element.value = '';
        alert('The uploaded file is not a JSON file.');
        return;
      }

      try {
        uploadedData = JSON.parse(await file.text());
        if (!isValid(uploadedData)) {
          element.value = '';
          uploadedData = [];
          throw Error('The uploaded file does not have a valid format');
        }
        d3.select('#file-name').text(file.name);
      } catch (err) {
        alert('The uploaded file does not have a valid format.');
      }
    });

  d3.select('#file-select')
    .on('click', () => {
      if (uploadedData.length === 0) return;
      data = uploadedData;
      [trainData, testData] = splitTrainTest(data);
      updatePoints();
      reset();
    });

  /* Main Column */


  /* Output Column */
  // Configure the number of trees
  // deleted code

  // Configure the max depth of each tree.
 

  // Configure the number of samples to train each tree.
  const percSamples = d3.select('#percSamples').on('input', function () {
    const element = this as HTMLInputElement;
    state.percSamples = +element.value;
    d3.select("label[for='percSamples'] .value")
      .text(element.value);
    reset();
  });
  percSamples.property('value', state.percSamples);
  d3.select("label[for='percSamples'] .value")
    .text(state.percSamples);

  const problem = d3.select('#problem').on('change', function () {
    state.problem = (problems as any)[(this as HTMLSelectElement).value]; //problems[(this as HTMLSelectElement).value];
    generateData();
    drawDatasetThumbnails();
    reset();
  });
  problem.property('value', getKeyFromValue(problems, state.problem));

  const showTestData = d3.select('#show-test-data').on('change', function () {
    state.showTestData = (this as HTMLInputElement).checked;
    state.serialize();
    //mainHeatMap.updateTestPoints(state.showTestData ? testData : []);
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property('checked', state.showTestData);

  const discretize = d3.select('#discretize').on('change', function () {
    state.discretize = (this as HTMLInputElement).checked;
    state.serialize();
    
  });
  // Check/uncheck the checbox according to the current state.
  discretize.property('checked', state.discretize);

  /* Data configurations */
  // Configure the ratio of training data to test data.
  const percTrain = d3.select('#percTrainData').on('input', function () {
    const element = this as HTMLInputElement;
    state.percTrainData = +element.value;
    d3.select("label[for='percTrainData'] .value")
      .text(element.value);
    reset();
  });
  percTrain.property('value', state.percTrainData);
  d3.select("label[for='percTrainData'] .value")
    .text(state.percTrainData);

  // Configure the level of noise.
  const noise = d3.select('#noise').on('input', function () {
    const element = this as HTMLInputElement;
    state.noise = +element.value;
    d3.select("label[for='noise'] .value")
      .text(element.value);
    generateData();
    reset();
  });
  const currentMax = parseInt(noise.property('max'));
  if (state.noise > currentMax) {
    if (state.noise <= 80) noise.property('max', state.noise);
    else state.noise = 50;
  } else if (state.noise < 0) state.noise = 0;
  noise.property('value', state.noise);
  d3.select("label[for='noise'] .value")
    .text(state.noise);

  /* Color map */
  // Add scale to the gradient color map.
  const x = d3
    .scaleLinear()
    .domain([-1, 1])
    .range([0, 144]);
  const xAxis = d3
    .axisBottom(x)
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format('d'));
  d3.select('#colormap g.core')
    .append('g')
    .attr('class', 'x axis')
    .attr('transform', 'translate(0,10)')
    .call(xAxis);
}

// 
function get2dArray(dataArray: Example2D[]) {
  if(dataArray != null && dataArray != undefined) {
    var resultMatrix: Number[][] = [];
    dataArray.forEach((i) => {
      resultMatrix.push([i.x, i.y]);
    });
    return resultMatrix;
  }
  return null;
}


function drawDatasetThumbnails() {
    const renderThumbnail = (canvas: any, dataGenerator: DataGenerator) => {
      const w = 100;
      const h = 100;
      canvas.setAttribute('width', w);
      canvas.setAttribute('height', h);
      const context = canvas.getContext('2d');
      const data = dataGenerator(200, 0);
      data.forEach((d: Example2D) => {
        context.fillStyle = colorScale(d.label);
        context.fillRect((w * (d.x + 6)) / 12, (h * (-d.y + 6)) / 12, 4, 4);
      });
      d3.select(canvas.parentNode).style('display', null);
    };
    d3.selectAll('.dataset').style('display', 'none');
  
    if (isClassification()) {
      for (const dataset in datasets) {
        const canvas: any = document.querySelector(
          `canvas[data-dataset=${dataset}]`
        );
        const dataGenerator = datasets[dataset];
        renderThumbnail(canvas, dataGenerator);
      }
    } else {
      for (const regDataset in regDatasets) {
        const canvas: any = document.querySelector(
          `canvas[data-regDataset=${regDataset}]`
        );
        const dataGenerator = regDatasets[regDataset];
        renderThumbnail(canvas, dataGenerator);
      }
    }
}

function generateData(firstTime = false) {
    if (!firstTime) {
      // Change the seed.
      state.seed = Math.random().toFixed(5);
      state.serialize();
    }
  
    seedrandom(state.seed);
  
    const numSamples = isClassification()
      ? NUM_SAMPLES_CLASSIFY
      : NUM_SAMPLES_REGRESS;
    const generator = isClassification()
      ? state.dataset
      : state.regDataset;
  
    data = generator(numSamples, state.noise / 100);
    // Shuffle the data in-place.
    shuffle(data);
    [trainData, testData] = splitTrainTest(data);
    updatePoints();
}

/**
 * Reset the app to initial state.
 * @param reset True when called on startup.
 */
function reset(onStartup = false) {
  if (!onStartup) {
    //trainWorker.terminate();
    isLoading(false);
  }

  options = {
    maxSamples: state.percSamples / 100,
    maxFeatures: 1.0,
    seed: undefined,
    useSampleBagging: true,
    replacement: false
  };

  if (isClassification()) {
    
    //TODO: 
    
  } else {
    
    //TODO:

  }

  trainMetrics = null;
  testMetrics = null;

  d3.select("#start-button .value")
    .text(isClassification() ? 'classify' : 'regress');


 // uploadedData = uploadedData || [];
  
  [trainData, testData] = splitTrainTest(data);

  state.serialize();
  updatePoints();
  updateUI(true);
}

/**
 * Split the input array into 2 chunks by an index determined by the selected
 * percentage of train data.
 * @param arr
 */
function splitTrainTest(arr: any[]): any[][] {
  const splitIndex = Math.floor((arr.length * state.percTrainData) / 100);
  return [arr.slice(0, splitIndex) , arr.slice(splitIndex)];
}

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
    d3.select('#output-heatmap canvas')
      .style('opacity', loading ? 0.2 : 1);
    d3.select('#output-heatmap svg')
      .style('opacity', loading ? 0.2 : 1); 
}

/**
 * Update all heat maps and metrics.
 * @param reset True when called in reset()
 */
function updateUI(reset = false) {

  // Metrics table
  d3.selectAll('.metrics tbody tr').remove();
  metricList.forEach((metric) => {
    const row = d3.select('.metrics tbody').append('tr');
    // First row contains metric name
    row.append('td')
      .attr('class', 'mdl-data-table__cell--non-numeric')
      .text(metric);
    // Next 2 rows contain train and test metric values
    row.append('td')
      .text(trainMetrics ? trainMetrics[metric].toFixed(3) : '0.000');
    row.append('td')
      .text(testMetrics ? testMetrics[metric].toFixed(3) : '0.000');
  });
}

function updatePoints() {
  mainHeatMap.updatePoints(trainData);
  mainHeatMap.updateTestPoints(state.showTestData ? testData : []);
  outputHeatMap.updatePoints(trainData);
}

function isClassification() {
    return state.problem === Problem.CLASSIFICATION;
}

function setClusterIndexes(clusters: number[]) {
  if (!!clusters && trainData.length == clusters.length) {
    trainData.forEach((val, idx) => {
      val.cluster = clusters[idx];
    });
  }
}

drawDatasetThumbnails();
makeGUI();
generateData(true);
reset(true);