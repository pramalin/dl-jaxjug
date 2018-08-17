package resources

import java.awt.image.BufferedImage

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.cpu.nativecpu.NDArray

import com.fasterxml.jackson.core.JsonParseException
import com.typesafe.config.ConfigFactory

import lib.Serialise
import lib.Utils
import play.api.libs.json.Json

object MNISTResources {

  lazy val model: MultiLayerNetwork = Serialise.read(ConfigFactory.load().getString("models.folder")+ConfigFactory.load().getString("models.model"))

  def parseData(in: String): Either[MNISTImage, String] = {
    val json = try {Some(Json.parse(in))} catch {
      case e: JsonParseException => None
    }
    val image = json.map(_ \ "image").flatMap(_.asOpt[Array[Double]]).map(MNISTImage)
    image match {
      case None => Right("Image not found/could not be parsed")
      case Some(x) if x.image.length != 28*28 => Right("Image was not 28x28")
      case Some(x) => Left(x)
    }
  }

  def recognise(image: MNISTImage): (Int, Array[Double]) = {
    val input = new NDArray(Utils.preProcess(image.toBufferedImage, 28, true).map(_.toFloat))

    val output = model.output(input, false).dup.data.asDouble
    var bestGuess = (-1d, -1)
    for ((x, i) <- output.zipWithIndex) {
      if (x > bestGuess._1) bestGuess = (x, i)
    }
    (bestGuess._2, output)
  }

}

case class MNISTImage(image: Array[Double]) {

  def toBufferedImage: BufferedImage = {
    val bufImage = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB)
    val intImage = image.map(X => (255 - X * 255).toInt)
    for (x <- Range(0, 28)) {
      for (y <- Range(0, 28)) {
        val value = intImage(y * 28 + x) << 16 | intImage(y * 28 + x) << 8 | intImage(y * 28 + x)
        bufImage.setRGB(x, y, value)
      }
    }
    bufImage
  }

}