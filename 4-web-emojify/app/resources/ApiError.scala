package resources

import play.api.libs.json.JsValue
import play.api.libs.json.Json

case class ApiError(error:String) {

  def toJson: JsValue = Json.obj(("error", Json.toJsFieldJsValueWrapper(error)))

}
