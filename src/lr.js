var mathjs = require("mathjs");
var randomPermutation = require("ml-util/randomPermutation");

var lrCostFunction = require("./lrCostFunction");
var oneVsAll = require("./oneVsAll");
var predictOneVsAll = require("./predictOneVsAll");

var XJSON = require("../data/X");
XJSON = mathjs.matrix(XJSON);

var yJSON = require("../data/y");
yJSON = mathjs.matrix(yJSON);

module.exports = function logisticRegression() {
    console.log("Starting Logistic Regression...");

    var inputLayerSize = 400;
    var numberOfLabels = 10;

    console.log("\nLoading data...\n");

    var m = XJSON._size[0];

    var randomIndices = randomPermutation(m);
    var selectedX = mathjs.subset(
        XJSON,
        mathjs.index(
            randomIndices.slice(0, 100),
            mathjs.range(0, XJSON._size[1])
        )
    );

    console.log("\nTesting lrCostFunction() "
        + "with regularization...\n");

    var theta_t = mathjs.matrix([
        [ -2 ],
        [ -1 ],
        [  1 ],
        [  2 ]
    ]);
    var X_t = mathjs.matrix([
        [ 1.00000,   0.10000,   0.60000,   1.10000 ],
        [ 1.00000,   0.20000,   0.70000,   1.20000 ],
        [ 1.00000,   0.30000,   0.80000,   1.30000 ],
        [ 1.00000,   0.40000,   0.90000,   1.40000 ],
        [ 1.00000,   0.50000,   1.00000,   1.50000 ]
    ]);
    var y_t = mathjs.matrix([
        [ 1 ],
        [ 0 ],
        [ 1 ],
        [ 0 ],
        [ 1 ]
    ]);
    var lambda_t = 3;

    var [ J, grad ] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

    console.log("\nCost: ", J, "\n");
    console.log("Expected cost: 2.534819\n");
    console.log("Gradients:\n");
    console.log(" ", grad, "\n");
    console.log("Expected gradients:\n");
    console.log(" 0.146561\n -0.548558\n 0.724722\n 1.398003\n");

    console.log("\nTraining One-vs-All Logistic Regression...\n");

    var lambda = 0.1;
    var allTheta = mathjs.ones(10, 401);
    allTheta = oneVsAll(XJSON, yJSON, numberOfLabels, lambda);
    var predictions = predictOneVsAll(allTheta, XJSON);
    var predictionsEqualY = mathjs.clone(yJSON);
    mathjs.map(
        predictionsEqualY,
        function predictionsEqualYCalculation(value, index, matrix) {
            predictionsEqualY = mathjs.subset(
                predictionsEqualY,
                mathjs.index(index[0], index[1]),
                value === predictions[index[0]] ? 1 : 0
            );
        }
    );
    console.log("Training Set Accuracy: ",
        mathjs.multiply(mathjs.mean(predictionsEqualY), 100));
    console.log("Logistic Regression Complete");
};
