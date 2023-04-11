"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const body_parser_1 = __importDefault(require("body-parser"));
const cors_1 = __importDefault(require("cors"));
const src_1 = require("./src");
const PORT = process.env.PORT || 4000;
const app = (0, express_1.default)();
const corsOptions = {
    origin: '*',
    optionsSuccessStatus: 200
};
app.use((0, cors_1.default)(corsOptions));
app.use(body_parser_1.default.json());
app.use(body_parser_1.default.urlencoded({ extended: true }));
app.use(src_1.DetectService);
app.use('/index.html', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    res.send(`
    <form enctype="multipart/form-data" method="post" action="/api/detect/predict">
      <input type="file" name="file">
      <button type="submit">Upload and predict</button>
    </form>
  `);
}));
app.get('/', (req, res) => {
    res.status(500).json({
        code: res.statusCode,
        status: false,
        message: 'Your endpoint is incorrect, please recheck abaout your endpoint..',
        env: process.env.NODE_ENV
    });
});
app.listen(PORT, () => {
    console.log(`Server is successfully running on port http://localhost:${PORT}`);
});
exports.default = app;
