const fs = require('fs');
const path = require('path');
const { parse } = require('csv-parse/sync');

class Parser {
    constructor(filePath) {
        this.filePath = path.resolve(__dirname, filePath);
    }

    readFile() {
        return fs.readFileSync(this.filePath, 'utf8');
    }

    parseCSV() {
        const fileContent = this.readFile();
        return parse(fileContent, {
            columns: true,
            skip_empty_lines: true,
            trim: true
        });
    }

    saveJSON(outputPath, data) {
        const resolvedPath = path.resolve(__dirname, outputPath);
        fs.writeFileSync(resolvedPath, JSON.stringify(data, null, 2));
    }
}

module.exports = Parser;