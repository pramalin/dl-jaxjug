package demo

import java.time.{Duration, Instant}
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._


object GpuDemo extends App {
  println("GPU demo")

  val a = Nd4j.rand(1, 1000000)
  val b = Nd4j.rand(1, 1000000)

  val t0 = Instant.now()
  val c = a.dot(b.T)

  println(s"c: $c")
  println(s"Vector dot operation took: ${Duration.between(t0, Instant.now()).toMillis()} milliseconds")


  val t1 = Instant.now()
  var d = 0.0
  for (i <- 0 to a.columns()  - 1) {
    d += a(i) * b(i)
  }

  println(s"loop d: $d")
  println(s"Traditional loop took: ${Duration.between(t1, Instant.now()).toMillis()} milliseconds")

}
