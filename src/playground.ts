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
    getKeyFromValue,
  } from './state';
import { DataGenerator, Example2D, shuffle, Get2dPoint, Point } from './dataset';
import * as utils from './utils';
import kmeans from 'ml-kmeans';
import * as dc from 'density-clustering';

// Number of samples in per dataset
const NUM_SAMPLES_CLASSIFY = 500;
// Size of the heatmaps.
const SIDE_LENGTH = 300;
// # of points per direction.
const DENSITY = 50;
  
const state = State.deserializeState();
const xDomain: [number, number] = [-6, 6];

const colorScale = d3
    .scaleLinear<string, number>()
    .domain([-1, 0, 1])
    .range(['#808B96', '#808B96', '#808B96'])
    .clamp(true);

// Plot the input heatmap.
const inputHeatMap = new HeatMap(
  SIDE_LENGTH,
  DENSITY,
  xDomain,
  xDomain,
  d3.select('#input-heatmap'),
  { showAxes: true }
);

// Plot the KMeans heatmap.
const kmeansHeatMap = new HeatMap(
  SIDE_LENGTH,
  DENSITY,
  xDomain,
  xDomain,
  d3.select('#kmeans-heatmap'),
  { showAxes: true }
);

// Plot the DB Scan heatmap.
const dbScanHeatMap = new HeatMap(
    SIDE_LENGTH,
    DENSITY,
    xDomain,
    xDomain,
    d3.select('#dbscan-heatmap'),
    { showAxes: true }
);



let data: Example2D[];
let uploadedData: Example2D[];
let testData: Example2D[];
let metricList = [];

/**
 * Prepares the UI on startup.
 */
function makeGUI() {
  d3.select('#start-button').on('click', () => {
    if((!validateEpsilon() && !validateNeighbors()) || !validateEpsilon() || !validateNeighbors()){
      return;
    }
    
    isLoading(true);
    //remove previous centroids
    clearCentroidData();

    // initializing input parameters for K Means method
    let inputData = get2dArray(testData);
    let noOfClusters = parseInt(state.clusters.toString());
  
    //**** Initial centres ****//
    let centroidIndexes = utils.randArray(0, testData.length-1, state.clusters);

    let centroidArray= [];
    centroidIndexes.forEach(item => {
        var cPt = testData[item];
        centroidArray.push([cPt.x,cPt.y]);
    });
    //**** Initial centres - ends ****//

    // K Means algorithm result
    let kmeansResult = kmeans(inputData, noOfClusters, { initialization: centroidArray});
    console.log(kmeansResult);
    // set index for resultant clusters 
    setClusterIndexes(kmeansResult.clusters);
    
    let mean_square_error = 0;
    let iterations = kmeansResult.iterations;
    kmeansResult.centroids.forEach((item, idx) => {
      //heighlight cluster centroids in color
      testData.push(Get2dPoint(item.centroid[0], item.centroid[1], 1, idx, true));
      mean_square_error += item.error;
    });

    //Show initial centroids in black color
    centroidIndexes.forEach(item => {
      var cPt = testData[item];
      testData.push(Get2dPoint(cPt.x, cPt.y, 1, 20, true)); 
    });
    // *** Display metrics ****//
    updateMetrics(false, mean_square_error,iterations);

    kmeansHeatMap.setColorScale(true);
    kmeansHeatMap.updatePoints(testData);

    let densityData = cloneInputData(inputData);
    DensityScan(densityData);

    isLoading(false);
  });
  // Select number clusters from dropdown data
  let centers = d3.select("#clusterCount").on("change", function() {
   state.clusters = (this as any).value;
   state.serialize();
  });

  /* Data column */
  d3.select('#data-regen-button').on('click', () => {
    generateData();
    reset();
  });

  const dataThumbnails = d3.selectAll('canvas[data-dataset]');
  dataThumbnails.on('click', function () {

    inputHeatMap.setColorScale(false);
    kmeansHeatMap.setColorScale(false);
    dbScanHeatMap.setColorScale(false);

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
//Formatting test dataset suitable for K Means method input
function get2dArray(dataArray: Example2D[]) {
  if(dataArray != null && dataArray != undefined) {
    var resultMatrix: number[][] = [];
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
  
    for (const dataset in datasets) {
      const canvas: any = document.querySelector(
        `canvas[data-dataset=${dataset}]`
      );
      const dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
}

function generateData(firstTime = false) {
    if (!firstTime) {
      // Change the seed.
      state.seed = Math.random().toFixed(5);
      state.serialize();
    }
  
    seedrandom(state.seed);
  
    const numSamples = NUM_SAMPLES_CLASSIFY
    const generator = state.dataset;
  
    data = generator(numSamples, state.noise / 100);
    // Shuffle the data in-place.
    shuffle(data);
    testData = data;
    updatePoints();
}

/**
 * Reset the app to initial state.
 * @param reset True when called on startup.
 */
function reset(onStartup = false) {
  if (!onStartup) {
    isLoading(false);
    inputHeatMap.setColorScale(false);
    kmeansHeatMap.setColorScale(false);
    dbScanHeatMap.setColorScale(false);
  }
  else {
    d3.select("#clusterCount").property('value', state.clusters);
  }
  
  testData = data;

  state.serialize();
  updatePoints();
  updateMetrics(true, null, null);
}

/**
 * Shows busy indicators in the UI as something is running in the background.
 * They include making all heatmaps opaque and showing a progress indicator next
 * to the cursor.
 * @param {boolean} loading True if something is running in the background
 */
function isLoading(loading: boolean) {
    d3.select('#input-heatmap canvas')
      .style('opacity', loading ? 0.2 : 1);
    d3.select('#input-heatmap svg')
      .style('opacity', loading ? 0.2 : 1); 
    d3.select('#kmeans-heatmap canvas')
      .style('opacity', loading ? 0.2 : 1);
    d3.select('#kmeans-heatmap svg')
      .style('opacity', loading ? 0.2 : 1); 
    d3.select('#dbscan-heatmap canvas')
      .style('opacity', loading ? 0.2 : 1);
    d3.select('#dbscan-heatmap svg')
      .style('opacity', loading ? 0.2 : 1);
    
}

/**
 * Update all heat maps and metrics.
 * @param reset True when called in reset()
 */
function updateMetrics(isReset: boolean, mse,iterations) {

  if(isReset) {
    d3.selectAll('.metrics table tr').remove();
    return;
  }

  // Metrics table
  let table = d3.select('.metrics table');
  table.selectAll('tr').remove();
  table.select('thead').append('tr').append('th').attr('colspan', '2').text('Metrics');
  let row1 = table.select('tbody').append('tr');
  row1.append('td').text('Mean Square Error');
  row1.append('td').text(mse);
  let row2 = table.select('tbody').append('tr');
  row2.append('td').text('Iterations');
  row2.append('td').text(iterations);
} 

function updatePoints() {
  inputHeatMap.updatePoints(testData);
  kmeansHeatMap.updatePoints(testData);
  dbScanHeatMap.updatePoints(testData);
}

function setClusterIndexes(clusters: number[]) {
  if (!!clusters && testData.length == clusters.length) {
    testData.forEach((val, idx) => {
      val.cluster = clusters[idx];
    });
  }
}

function clearCentroidData() {
  if (testData != undefined && testData.length > 0) {
    testData = testData.filter(i => i.IsCentroid == false);
  }
}

function DensityScan(inputData: number[][]) {
  let dbscan = new dc.DBSCAN();
 
  let epsilon = d3.select('.txt_epsilon').property("value");
  let pointsInNeighborhood = d3.select('.txt_neighborPoints').property("value");
  console.log(epsilon);
  console.log(pointsInNeighborhood);

  let clusters = dbscan.run(inputData, epsilon, pointsInNeighborhood);
  let dbScanPoints: Example2D[] = [];

  let finalArrList = [];
  clusters.forEach((arr, idx) => {
    finalArrList = finalArrList.concat(arr);
    arr.forEach(pt => {
      var cPt = testData[pt];
      dbScanPoints.push(Get2dPoint(cPt.x, cPt.y, cPt.label, idx, false));
    });
  });

  let missingData = inputData.filter((item, idx) => {
    return !finalArrList.some(d => d == idx);
  });

  if (missingData != undefined && missingData.length > 0) {
    missingData.forEach(cPt => {
      dbScanPoints.push(Get2dPoint(cPt[0], cPt[1], 1, 20, false));
    });
  }

  shuffle(dbScanPoints);

  dbScanHeatMap.setColorScale(true);
  dbScanHeatMap.updatePoints(dbScanPoints);
}

function cloneInputData(inputData: number[][]) {
  let result: number[][] = [];
  inputData.forEach(item => {
    result.push([item[0], item[1]]);
  });

  return result;
}

function validateEpsilon() {
  let txtEpsilon = d3.select('.txt_epsilon').property("value");
  let epsilon = parseFloat(txtEpsilon);

  if(isNaN(epsilon) || epsilon < 0.1 || epsilon > 3.0) {
    d3.select('.txt_epsilon').style("border", "1px solid red");
    return false;
  }
  else {
    d3.select('.txt_epsilon').style("border", "1px solid rgb(118, 118, 118)");
    return true;
  }
}

function validateNeighbors() {
  let txtNeighbors = d3.select('.txt_neighborPoints').property("value");
  let neighbors = parseInt(txtNeighbors);

  if(isNaN(neighbors) || neighbors < 2 || neighbors > 100) {
    d3.select('.txt_neighborPoints').style("border", "1px solid red");
    return false;
  }
  else {
    d3.select('.txt_neighborPoints').style("border", "1px solid rgb(118, 118, 118)");
    return true;
  }
}

d3.select('.txt_epsilon').on('input', function(eventArgs) {
  let textbox = <any>this;
  textbox.value = textbox.value.replace(/[^0-9.]/g, '').replace(/(\..*?)\..*/g, '$1');

  let epsilon = parseFloat(textbox.value);

  if(isNaN(epsilon) || epsilon < 0.1 || epsilon > 3.0) {
    d3.select('.txt_epsilon').style("border", "1px solid red");
    return;
  }
  else {
    d3.select('.txt_epsilon').style("border", "1px solid rgb(118, 118, 118)");
  }
});

d3.select('.txt_neighborPoints').on('input', function(eventArgs) {
  let textbox = <any>this;
  textbox.value = textbox.value.replace(/[^0-9]/g, '').replace(/(\..*?)\..*/g, '$1');

  let neighbors = parseInt(textbox.value);

  if(isNaN(neighbors) || neighbors < 2 || neighbors > 100) {
    d3.select('.txt_neighborPoints').style("border", "1px solid red");
    return;
  }
  else {
    d3.select('.txt_neighborPoints').style("border", "1px solid rgb(118, 118, 118)");
  }
});

drawDatasetThumbnails();
makeGUI();
generateData(true);
reset(true);