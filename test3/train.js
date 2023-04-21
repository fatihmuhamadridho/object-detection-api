// Import TensorFlow.js dan model coco-ssd
const tf = require('@tensorflow/tfjs');
const cocossd = require('@tensorflow-models/coco-ssd');

// Buat fungsi async untuk melatih model dengan gambar dan COCO annotations
async function trainModel() {
  // Parsing file JSON COCO format ke dalam JavaScript object.
  const cocoData = {
    info: {
      year: 2023,
      version: '1.0',
      description:
        'VIA project exported to COCO format using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)',
      contributor: '',
      url: 'http://www.robots.ox.ac.uk/~vgg/software/via/',
      date_created: 'Thu Apr 20 2023 23:00:26 GMT+0700 (Waktu Indonesia Barat)'
    },
    images: [
      {
        id: 1,
        width: 600,
        height: 400,
        file_name: 'adutta_swan.jpg',
        license: 0,
        flickr_url: 'adutta_swan.jpg',
        coco_url: 'adutta_swan.jpg',
        date_captured: ''
      },
      {
        id: 2,
        width: 600,
        height: 394,
        file_name: 'wikimedia_death_of_socrates.jpg',
        license: 0,
        flickr_url: 'wikimedia_death_of_socrates.jpg',
        coco_url: 'wikimedia_death_of_socrates.jpg',
        date_captured: ''
      }
    ],
    annotations: [
      {
        segmentation: [
          [
            116, 157, 94, 195, 176, 264, 343, 273, 383, 261, 385, 234, 369, 222, 406, 216, 398, 155,
            364, 124, 310, 135, 297, 170, 304, 188, 244, 170, 158, 175
          ]
        ],
        area: 46488,
        bbox: [94, 124, 312, 149],
        iscrowd: 0,
        id: 1,
        image_id: 1,
        category_id: 1
      },
      {
        segmentation: [[174, 139, 282, 139, 282, 366, 174, 366]],
        area: 24516,
        bbox: [174, 139, 108, 227],
        iscrowd: 0,
        id: 2,
        image_id: 2,
        category_id: 2
      },
      {
        segmentation: [[347, 114, 438, 114, 438, 323, 347, 323]],
        area: 19019,
        bbox: [347, 114, 91, 209],
        iscrowd: 0,
        id: 3,
        image_id: 2,
        category_id: 2
      },
      {
        segmentation: [
          [
            333, 180, 332.935, 181.046, 332.742, 182.084, 332.421, 183.106, 331.975, 184.104,
            331.407, 185.071, 330.722, 186, 329.926, 186.883, 329.023, 187.713, 328.021, 188.485,
            326.927, 189.193, 325.751, 189.83, 324.5, 190.392, 323.185, 190.876, 321.814, 191.276,
            320.4, 191.591, 318.952, 191.818, 317.482, 191.954, 316, 192, 314.518, 191.954, 313.048,
            191.818, 311.6, 191.591, 310.186, 191.276, 308.815, 190.876, 307.5, 190.392, 306.249,
            189.83, 305.073, 189.193, 303.979, 188.485, 302.977, 187.713, 302.074, 186.883, 301.278,
            186, 300.593, 185.071, 300.025, 184.104, 299.579, 183.106, 299.258, 182.084, 299.065,
            181.046, 299, 180, 299.065, 178.954, 299.258, 177.916, 299.579, 176.894, 300.025,
            175.896, 300.593, 174.929, 301.278, 174, 302.074, 173.117, 302.977, 172.287, 303.979,
            171.515, 305.073, 170.807, 306.249, 170.17, 307.5, 169.608, 308.815, 169.124, 310.186,
            168.724, 311.6, 168.409, 313.048, 168.182, 314.518, 168.046, 316, 168, 317.482, 168.046,
            318.952, 168.182, 320.4, 168.409, 321.814, 168.724, 323.185, 169.124, 324.5, 169.608,
            325.751, 170.17, 326.927, 170.807, 328.021, 171.515, 329.023, 172.287, 329.926, 173.117,
            330.722, 174, 331.407, 174.929, 331.975, 175.896, 332.421, 176.894, 332.742, 177.916,
            332.935, 178.954
          ]
        ],
        area: 816,
        bbox: [299, 168, 34, 24],
        iscrowd: 0,
        id: 4,
        image_id: 2,
        category_id: 3
      }
    ],
    licenses: [{ id: 0, name: 'Unknown License', url: '' }],
    categories: [
      { supercategory: 'type', id: 1, name: 'Bird' },
      { supercategory: 'type', id: 2, name: 'Human' },
      { supercategory: 'type', id: 3, name: 'Cup (object)' },
      { supercategory: 'type', id: 4, name: 'Unknown (object)' }
    ]
  };

  // Membuat model TensorFlow.js dan menambahkan layer-layer yang diperlukan.
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({ inputShape: [600, 400, 3], filters: 16, kernelSize: 3, activation: 'relu' })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));

  // Melatih model menggunakan data COCO.
  const images = cocoData.images.map((image) => {
    return tf.browser.fromPixels(`'./images/${image.file_name}'`);
  });
  const annotations = cocoData.annotations.map((annotation) => {
    return {
      x: annotation.bbox[0],
      y: annotation.bbox[1],
      width: annotation.bbox[2],
      height: annotation.bbox[3],
      label: annotation.category_id
    };
  });
  const xs = tf.stack(images);
  const ys = tf.oneHot(
    tf.tensor1d(
      annotations.map((annotation) => annotation.label),
      'int32'
    ),
    4
  );
  model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  model.fit(xs, ys, { epochs: 10 }).then(async () => {
    // Mengonversi model ke dalam format yang dapat digunakan oleh TensorFlow.js.
    const converter = tf.savedModelConverter;
    const tfjsModel = await converter.convert(model);
    await tfjsModel.save('./models');
  });
}

// Panggil fungsi trainModel
trainModel();
