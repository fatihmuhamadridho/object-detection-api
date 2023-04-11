import { default as express } from 'express';
import { default as bodyParser } from 'body-parser';
import { default as cors } from 'cors';
import { DetectService } from './src';

const PORT = process.env.PORT || 4000;
const app = express();

const corsOptions = {
  origin: '*',
  optionsSuccessStatus: 200
};

app.use(cors(corsOptions));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.use(DetectService);

app.use('/index.html', async (req: any, res: any) => {
  res.send(`
    <form enctype="multipart/form-data" method="post" action="/api/detect/predict">
      <input type="file" name="file">
      <button type="submit">Upload and predict</button>
    </form>
  `);
});

app.get('/', (req: any, res: any) => {
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

export default app;
