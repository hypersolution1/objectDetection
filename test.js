
process.env['TF_CPP_MIN_LOG_LEVEL'] = '3'

var cv = require('opencv4nodejs')

var fs = require('fs')
var readFileAsync = require('util').promisify(fs.readFile)

var objdetect = require('.')


;(async () => {

  await objdetect.init()

  var imgsrc = await cv.imreadAsync(`/var/store/test/truck1-1.jpg`)

  var objs
  console.time('time')
  objs = await objdetect.detect(imgsrc)
  console.timeEnd('time')

  console.log(objs)

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
