var cv = require('opencv4nodejs')
var sharp = require('sharp')

var fs = require('fs')
var readFileAsync = require('util').promisify(fs.readFile)

var TFModel = require('libtf')

var model

var labels = {}

var close = exports.close = () => {
  //
}
//process.on('exit', close)

exports.init = async function (imgsrc) {
  model = TFModel({
    'gpu_memory_fraction': 0.10,
  })  
  //await model.load(`${__dirname}/models/ssdlite_mobilenet_v2_coco_2018_05_09.pb`)
  await model.load(`${__dirname}/models/ssd_inception_v2_coco_2018_01_28.pb`)
  var labelmap = JSON.parse(await readFileAsync(`${__dirname}/models/coco_labelmap.json`))
  for(var l of labelmap) {
    labels[l.id] = l.display_name
  }  
}

async function pad_image(img, padheight, padwidth) {
  var height = img.rows
  var width = img.cols
  var channel = img.channels

  if(height > padheight || width > padwidth) {
    width = (width > padwidth) ? padwidth : width
    height = (height > padheight) ? padheight : height
    img = await img.getRegion(new cv.Rect(0, 0, width, height)).copyAsync()
  }
  if(height < padheight || width < padwidth) {
    var img = await img.copyMakeBorderAsync(0, padheight - height, 0, padwidth - width, cv.BORDER_CONSTANT, new cv.Vec3(0, 0, 0))
  }
 
  return img
}

async function imgResize(img, width, height) {
  var sharpresize = await sharp(img.getData(), {
    raw: {
        width: img.cols,
        height: img.rows,
        channels: 3
    }
  })
  .resize({ width, height })
  .toBuffer({ resolveWithObject: true })
  return new cv.Mat(sharpresize.data, sharpresize.info.height, sharpresize.info.width, cv.CV_8UC3)
}

exports.detect = async function (img) {

  var ratio = img.cols / img.rows
  var imginput = await imgResize(img, 300)

  var input = {
    "image_tensor": {
      "dim": [1, imginput.rows, imginput.cols, 3],
      "data": imginput.getData(),
    },
    
  }

  var val = await model.execute(input, ["detection_boxes", "detection_scores", "num_detections", "detection_classes"])

  var objs = []

  var num_detections = val.num_detections.data[0]
  for (var i = 0; i < num_detections; i++) {
    var score = val.detection_scores.data[i]
    if(score > 0.3) {
      var detection_boxes = val.detection_boxes.data.subarray(4*i,4*(i+1))
      var [ymin, xmin, ymax, xmax] = detection_boxes
      var label = labels[val.detection_classes.data[i]]
      objs.push({
        label,
        score,
        ptmin: [xmin*img.cols, ymin*img.rows],
        ptmax: [xmax*img.cols, ymax*img.rows],
      })
    }
  }
  return objs

}


