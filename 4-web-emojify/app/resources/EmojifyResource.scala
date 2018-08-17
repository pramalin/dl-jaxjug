package resources

import com.typesafe.config.ConfigFactory
import lib.Serialise
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import services.EmojifyService

import scala.io.Source


object EmojifyResource {
  val emojiService = new EmojifyService

  def suggestedEmoji(text: String): (String) = {
    emojiService.suggestedEmoji(text)
  }

  def main(args: Array[String]): Unit = {
    val text = "she got me a nice present"
    println(s"Text: $text suggested emoji : ${suggestedEmoji(text)}")
  }
}