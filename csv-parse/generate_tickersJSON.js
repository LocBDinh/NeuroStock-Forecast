/*

Script to generate tickers.json, used in 
front end validation of ticker inputs / 
autocomplete search bar

*/

const fs = require("fs");
const { parse } = require("csv-parse");

const jsonArray = [];
fs.createReadStream("./ticker_list.csv");
fs.createReadStream("./ticker_list.csv")
.pipe(parse({ delimiter: ",", from_line: 2 }))
.on("data", function (row) {
    const jsonObject = {
        ticker: row[0],
        name: row[1]
    };
    jsonArray.push(jsonObject);
})
.on("error", function (error) {
    console.log(error.message);
})
.on("end", function () {
    const jsonString = JSON.stringify(jsonArray, null, 2);
    fs.writeFile('tickers.json', jsonString, 'utf8', function(err) {
        if (err) {
            console.error('Error writing to file:', err);
        } else {
            console.log('JSON object has been written to output.json');
        }
    });
});


