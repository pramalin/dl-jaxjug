package controllers

import play.api.libs.json.{JsValue, Json}
import play.api.mvc.{Action, Controller}
import resources.{ApiError, EmojifyResource}
import lib.Serialise
import com.typesafe.config.ConfigFactory
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

class EmojifyEndpoint extends Controller {

  def index = Action {

    Ok(views.html.emojify.emojify_home())

  }

  def suggest = Action { implicit request =>

    val body = request.body.asFormUrlEncoded.get("body").head
    val text = (Json.parse(body) \ "text").get
    val emoji = EmojifyResource.suggestedEmoji(text.as[String])

    Ok (Json.prettyPrint(Json.obj("text" -> text, "suggested" -> emoji)))
  }

}
