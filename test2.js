
process.env['TF_CPP_MIN_LOG_LEVEL'] = '3'

var cv = require('opencv4nodejs')

var fs = require('fs')
var readFileAsync = require('util').promisify(fs.readFile)

var ObjDetect = require('.')

;(async () => {

  var config = {
    'gpu_memory_fraction': 0.10,
    //'minimum_score': 0.3,
    //'model': 'test/GPU_test_body_Ec2_July',
    //'model': 'ssdlite_mobilenet_v2_coco_2018_05_09',
  }

  //var objdetect = await ObjDetect(config)
  var objdetects = []
  var init = 1
  while(init--) {
    objdetects.push(await ObjDetect(config))
  }

  //var imgsrc = await cv.imreadAsync(`/var/store/test/truck1-1.jpg`)
  var imgsrc = await cv.imreadAsync(`/var/store/test/00001630.jpg`)

  var objs

  //objs = await objdetect.detect(imgsrc)
  objs = (await Promise.all(objdetects.map((objdetect) => {
    return objdetect.detect(imgsrc)
  })))[0]

  var meantime = 0
  var cnt = 10
  while(cnt--) {
    var hrstart = process.hrtime()
    //console.time('time')
    //objs = await objdetect.detect(imgsrc)
    objs = (await Promise.all(objdetects.map((objdetect) => {
      return objdetect.detect(imgsrc)
    })))[0]
    //console.timeEnd('time')
    var [,hrend] = process.hrtime(hrstart)
    meantime += hrend / 1000000
    meantime /= 2
  }
  console.log("Execution time:", Math.round(meantime), "ms,", Math.round(1000 / meantime), "fps")

  //console.log(objs)

  for(let obj of objs) {
    var score = (obj.score*100).toFixed(0)
    imgsrc.drawRectangle(new cv.Point2(obj.ptmin[0], obj.ptmin[1]), 
      new cv.Point2(obj.ptmax[0], obj.ptmax[1]), new cv.Vec3(0, 255, 0), 4)
    imgsrc.putText(`${obj.label} ${score}%`,
      new cv.Point2(obj.ptmin[0], obj.ptmin[1] + 30), cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Vec3(255, 255, 255), 5)      
    imgsrc.putText(`${obj.label} ${score}%`,
      new cv.Point2(obj.ptmin[0], obj.ptmin[1] + 30), cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Vec3(0, 0, 0), 2) 
  }
  await cv.imwriteAsync("out.jpg", imgsrc)

})()
.catch(function (err) {
  console.log(err)
})
