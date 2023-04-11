import { DetectController } from './controller';
import { Router } from 'express';

import * as cocoSsd from '@tensorflow-models/coco-ssd';
import multer from 'multer';

const DetectService = Router();
const upload = multer({ dest: 'public/uploads' });

let model: any;
cocoSsd.load().then((loadedModel: any) => {
  model = loadedModel;
});

DetectService.get('/api/detect/test', async (req: any, res: any) => {
  try {
    const response = await DetectController.index();
    res.status(200).json(response);
  } catch (error: any) {
    res.status(500).json({ status: res.statusCode, error: error });
  }
});

DetectService.post('/api/detect/predict', upload.single('file'), async (req: any, res: any) => {
  try {
    const { path: filePath } = req.file;
    const fileType = req.file.mimetype.split('/')[0];

    if (fileType !== 'image' && fileType !== 'video') {
      return res.status(400).send('Invalid file type');
    }

    switch (fileType) {
      case 'image':
        const responseImage = await DetectController.create.uploadImage(filePath);
        return res
          .status(200)
          .json({ status: true, message: 'File uploaded successfully', data: responseImage });
      case 'video':
        await DetectController.create.uploadVideo(filePath);
        return res
          .status(200)
          .json({ status: true, message: 'File uploaded successfully', data: [] });
      default:
        return res.status(400).send({ status: false, message: 'Invalid file type' });
    }
  } catch (error: any) {
    res.status(500).json({ status: res.statusCode, error: error });
  }
});

export { DetectService };
