var mathjs = require("mathjs");

module.exports = function (allTheta, X) {
    var m = X._size[0];
    var numberOfLabels = allTheta._size[0];

    var onesRow = mathjs.ones(X._size[0], 1);
    X = mathjs.concat(onesRow, X, 1);

    var XProductAllThetaTranspose = mathjs.multiply(X, mathjs.transpose(allTheta));

    var maxIndices = {};
    mathjs.map(XProductAllThetaTranspose, function (value, index, matrix) {
        if(!maxIndices[index[0]]) {
            maxIndices[index[0]] = {
                value: value,
                index: 1
            }
        }
        if(maxIndices[index[0]].value < value) {
            maxIndices[index[0]] = {
                value: value,
                index: index[1] + 1
            };
        }
    });

    var sumOfMaximums = 0;
    var sumOfIndices = 0;

    for(var key in maxIndices) {
        sumOfMaximums += maxIndices[key].value;
        sumOfIndices += maxIndices[key].index;
    }

    var indexList = [];
    for(var i in maxIndices) {
        indexList.push(maxIndices[i].index);
    }
    return indexList;
};

