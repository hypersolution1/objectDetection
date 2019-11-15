
var sharp = require('sharp')

var fs = require('fs')
var readFileAsync = require('util').promisify(fs.readFile)

var TFModel = require('libtf')

var options

module.exports = async function (opt = {}) {

  options = Object.assign({
    'gpu_memory_fraction': 0.10,
    'minimum_score': 0.3,
    'model': 'ssd_inception_v2_coco_2018_01_28',
  }, opt)

  var model = TFModel({
    'gpu_memory_fraction': options.gpu_memory_fraction,
  })  
  //await model.load(`${__dirname}/models/ssdlite_mobilenet_v2_coco_2018_05_09.pb`) // ssd_inception_v2_coco_2018_01_28
  await model.load(`${__dirname}/models/${options.model}.pb`)
  var labelmap = JSON.parse(await readFileAsync(`${__dirname}/models/coco_labelmap.json`))
  var labels = {}
  for(var l of labelmap) {
    labels[l.id] = l.display_name
  }
  
  // async function pad_image(img, padheight, padwidth) {
  //   var height = img.rows
  //   var width = img.cols
  //   var channel = img.channels

  //   if(height > padheight || width > padwidth) {
  //     width = (width > padwidth) ? padwidth : width
  //     height = (height > padheight) ? padheight : height
  //     img = await img.getRegion(new cv.Rect(0, 0, width, height)).copyAsync()
  //   }
  //   if(height < padheight || width < padwidth) {
  //     var img = await img.copyMakeBorderAsync(0, padheight - height, 0, padwidth - width, cv.BORDER_CONSTANT, new cv.Vec3(0, 0, 0))
  //   }
  
  //   return img
  // }

  async function imgResize(img, width, height) {
    var sharpresize = await sharp(await img.getDataAsync(), {
      raw: {
          width: img.cols,
          height: img.rows,
          channels: 3
      }
    })
    .resize({ width, height })
    .toBuffer({ resolveWithObject: true })
    return {
      data: sharpresize.data, 
      height: sharpresize.info.height, 
      width: sharpresize.info.width, 
    }
  }

  var detect = async function (img) {
    
    var ratio = img.cols / img.rows
    var imginput = await imgResize(img, 300)

    var input = {
      "image_tensor": {
        "dim": [1, imginput.height, imginput.width, 3],
        "data": imginput.data,
      },
      
    } 

    var val = await model.execute(input, ["detection_boxes", "detection_scores", "num_detections", "detection_classes"])

    var objs = []

    var num_detections = val.num_detections.data[0]
    for (var i = 0; i < num_detections; i++) {
      var score = val.detection_scores.data[i]
      if(score > options.minimum_score) {
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

  return {
    detect
  }
}
