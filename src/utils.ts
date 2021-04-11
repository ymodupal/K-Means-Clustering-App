import * as d3 from 'd3';

const NUM_SHADES = 30;

// Get a range of colors.
const tmpScale = d3
  .scaleLinear<string, number>()
  .domain([0, 0.5, 1])
  .range(['#f59322', '#e8eaeb', '#0877bd','#AB3144','#2DB83F','#29A3C1','#A74FC0'])
  .clamp(true);
 // red color - #F44336
 

// Due to numerical error, we need to specify
// d3.range(0, end + small_epsilon, step)
// in order to guarantee that we will have end/step entries with
// the last element being equal to end.
const colors = d3
  .range(0, 1 + 1e-9, 1 / NUM_SHADES)
  .map((a) => tmpScale(a));

const color = d3
  .scaleQuantize()
  .domain([-1, 1])
  .range(colors);

/**
 * Computes the average mean square error of the predicted values.
 * @param yPred Estimated target value.
 * @param yTrue Correct target value.
 */
function getLoss(yPred: number[], yTrue: number[]): number {
  if (yPred.length !== yTrue.length) {
    throw Error('Length of predictions must equal length of labels');
  }
  let loss = 0;
  for (let i = 0; i < yPred.length; i++) {
    loss += 0.5 * (yPred[i] - yTrue[i]) ** 2;
  }
  return loss / yPred.length;
}

// Generating random indexes for centroids
// Assigning generated numbers to an array
function randArray(a, b, clustercount) {
  var resultArray = [];

  while (resultArray.length < clustercount) {
    var randInt = getRandomInt(a, b);
    if (resultArray.indexOf(randInt) === -1) {
      resultArray.push(randInt);
    }
  }

  return resultArray;
}
// function to generate a random number
function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export {
  color,
  getLoss,
  randArray,
  getRandomInt
};
