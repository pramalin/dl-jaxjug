package lib

import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import java.io.File

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.DenseVector
import javax.imageio.ImageIO

object Utils {

  def readMinst(offset: Int = 0, n: Int = 0): List[(Int, DenseVector[Double])] = {

    val bufferedSource = scala.io.Source.fromFile("data/mnist.csv")

    val data: ArrayBuffer[(Int, DenseVector[Double])] = ArrayBuffer()

    val iterator = bufferedSource.getLines().drop(offset)

    for (line <- if (n == 0) iterator else iterator.take(n)) {
      val cols = line.split(",").map(_.trim)

      val label = cols(0).toInt
      val vect = ArrayBuffer[Double]()
      for (i <- Range(0, 28*28)) {
        //TODO: Check that this loads correct data
        vect.append(
          (cols(i + 1).toDouble/255) * 2 - 1d
        )
      }

      data.append((label, DenseVector[Double](vect.toArray)))

    }

    bufferedSource.close()

    data.toList

  }

  def writeText(text: String, path: String): Unit = {
    import java.io._
    val pw = new PrintWriter(new File(path))
    pw.write(text)
    pw.close()
  }

  def readText(path: String): String = {
    val s = scala.io.Source.fromFile(path)
    s.getLines().toList.mkString("\n")
  }

  def preProcess(image: BufferedImage, size: Int, log: Boolean = false): Array[Double] = {
    val imageArray = Array.ofDim[Double](size, size)

    val rotatedImage = new BufferedImage(size,size,image.getType())
    val g = rotatedImage.createGraphics()
    g.rotate(Math.toRadians(90), size/2,size/2)

    g.drawImage(image, 0, size, size , -size, null)

    val finalImage = new ArrayBuffer[Double]()

    var xtotal, ytotal = 0d
    var num = 0d

    for (x <- Range(0,size)) {
      for (y <- Range(0, size)) {

        val p = rotatedImage.getRGB(x, y)
        val a = 255 - ((p >> 24) & 0xff)
        val r = 255 - ((p >> 16) & 0xff)
        val g = 255 - ((p >> 8) & 0xff)
        val b = p & 0xff


        rotatedImage.setRGB(x, y, (a<<24) | (r<<16) | (g<<8) | b)
        xtotal += x * r
        ytotal += y * r
        num += r
      }
    }

    val com_1 = (xtotal/num, ytotal/num)
    if (log) println(f"Old Centre of mass: (${com_1._1}%1.2f, ${com_1._2}%1.2f)")

    val translatedImage = new BufferedImage(size,size,image.getType())
    val g_translate = translatedImage.createGraphics()

    val translate = ((14 - com_1._1) /2, (14 - com_1._2)/2)
    if (log) println(f"Translate by (${translate._1}%1.2f,${translate._2}%1.2f)")

    val tx: AffineTransform = new AffineTransform
    tx.translate(translate._1, translate._2)
    g_translate.setTransform(tx)
    g_translate.drawImage(rotatedImage, tx, null)

    xtotal = 0d
    ytotal = 0d
    num = 0d

    for (x <- Range(0,size)) {
      for (y <- Range(0, size)) {
        val colour = translatedImage.getRGB(x, y)
        val r = ((colour & 0xff0000) >> 16) / 255d
        finalImage.append(
          if (r < 0.02) 0
          else r
        )
        xtotal += x * r
        ytotal += y * r
        num += r
      }
    }

    val com_2 = (Math.round(xtotal/num), Math.round(ytotal/num))
    if (log) println(f"New Centre of mass: (${com_2._1}, ${com_2._2}%1.2f)\n")

    finalImage.toArray
  }


  def readBWBMP(path: String, size: Int, log: Boolean = false): Array[Double] = {
    if (log) println(s"Loading image $path")
    val image = ImageIO.read(new File(path))
    preProcess(image, size,log)
  }

}
