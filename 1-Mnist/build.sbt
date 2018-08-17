name := "mnist"
organization := "tutorial"
version := "0.1"

scalaVersion := "2.11.8"

classpathTypes += "maven-plugin"

val nd4jVersion = "0.9.1"

libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0-platform" % nd4jVersion
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-cuda-8.0" % nd4jVersion
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % nd4jVersion
libraryDependencies += "ch.qos.logback"     %  "logback-classic" % "1.1.7"
