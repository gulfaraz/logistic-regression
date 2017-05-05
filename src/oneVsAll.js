var mathjs = require("mathjs");

module.exports = function (X, y, numberOfLabels, lambda) {
    var m = X._size[0];
    var n = X._size[1];
    var allTheta = [];
    var X = mathjs.concat(mathjs.ones([ m, 1 ]), X);
    var initialTheta = mathjs.zeros([ n + 1, 1 ]);
    var fmincgOptions = {
        "maxIterations" : 100
    };

    var lrCostFunction = require("./lrCostFunction");

    function lrCostFunctionClosure(X, yEqualC, lambda) {
        return (function (theta) {
            return lrCostFunction(theta, X, yEqualC, lambda);
        });
    }

    var fmincg = require("ml-util/fmincg");
    for(var i=1; i<=numberOfLabels; i++) {
        console.log("Training Label - " + i + "...");
        var yEqualC = mathjs.clone(y);
        mathjs.map(yEqualC, function (value, index, matrix) {
            yEqualC = mathjs.subset(yEqualC, mathjs.index(index[0], index[1]), value === i ? 1 : 0 );
        });

        var optimizedValue = fmincg(lrCostFunctionClosure(X, yEqualC, lambda), initialTheta, fmincgOptions);
        allTheta.push(mathjs.flatten(mathjs.transpose(optimizedValue[0])));
    }
    allTheta = mathjs.matrix(allTheta);
    return allTheta;
};

