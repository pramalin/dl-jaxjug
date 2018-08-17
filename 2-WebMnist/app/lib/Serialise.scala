package lib

import java.io.File

import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer

object Serialise {

  def save(path: String, model:Model) = {
    val location = new File(path)
    ModelSerializer.writeModel(model, location, true)
  }

  def read(path: String): MultiLayerNetwork = {
    ModelSerializer.restoreMultiLayerNetwork(path)
  }

}
