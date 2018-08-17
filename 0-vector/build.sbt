name := "0-vector"
organization := "tutorial"
version := "1.0"

scalaVersion := "2.10.6"
val nd4jVersion = "0.9.1"

libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0-platform" % nd4jVersion
libraryDependencies += "org.nd4j" %% "nd4s" % nd4jVersion
libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.1.7"
