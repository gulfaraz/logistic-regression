var mathjs = require("mathjs");
var sigmoid = require("ml-util/sigmoid");

module.exports = function logisticRegressionCostFunction(
    theta,
    X,
    y,
    lambda
) {
    var m = y._size[0];
    var J = 0;
    var grad = [];

    for(var i=0; i<theta.length; i++) {
        grad.push(
            Array.apply(
                null,
                Array(theta[0].length)
            ).map(
                Number.prototype.valueOf,
                0
            )
        );
    }

    var hx = sigmoid(mathjs.multiply(X, theta));

    var newTheta = mathjs.clone(theta);
    newTheta = mathjs.subset(newTheta, mathjs.index(0, 0), 0);

    var yTranspose = mathjs.transpose(y);
    var hxLog = mathjs.log(hx);
    var ySubtractOne = mathjs.subtract(1, y);
    var ySubtractOneTranspose = mathjs.transpose(ySubtractOne);
    var hxSubtractOne = mathjs.subtract(1, hx);
    var hxSubtractOneLog = mathjs.log(hxSubtractOne);
    var yTransposeProductHxLog = mathjs.multiply(yTranspose, hxLog);
    var ySubtractOneTransposeProductHxSubtractOneLog
        = mathjs.multiply(ySubtractOneTranspose, hxSubtractOneLog);
    var jFactor1 = mathjs.multiply(
        -1/m,
        mathjs.add(
            yTransposeProductHxLog,
            ySubtractOneTransposeProductHxSubtractOneLog
        )
    );
    var jFactor2 = mathjs.multiply(
        (lambda/(2*m)),
        mathjs.sum(
            mathjs.multiply(
                mathjs.transpose(newTheta),
                newTheta
            )
        )
    );
    J = mathjs.sum(mathjs.add(jFactor1, jFactor2));

    var hxSubtractYTransposeProductX = mathjs.multiply(
        mathjs.transpose(
            mathjs.subtract(hx, y)
        ),
        X
    );
    var gradFactor1 = mathjs.multiply(
        1/m,
        hxSubtractYTransposeProductX
    );
    var gradFactor2 = mathjs.multiply(
        lambda/m,
        mathjs.transpose(newTheta)
    );
    grad = mathjs.add(gradFactor1, gradFactor2);
    grad = mathjs.flatten(grad);
    return [ J, grad ];
};
