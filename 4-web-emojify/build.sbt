name := "4-WebEmojify"

version := "1.0"

lazy val `webmachinelearning` = (project in file(".")).enablePlugins(PlayScala, LauncherJarPlugin)

scalaVersion := "2.11.11"

// EclipseKeys.preTasks := Seq(compile in Compile, compile in Test)

libraryDependencies ++= Seq( jdbc , cache , ws   , specs2 % Test )

unmanagedResourceDirectories in Test <+=  baseDirectory ( _ /"target/web/public/test" )

// unmanagedBase := baseDirectory.value / "lib"

libraryDependencies  ++= Seq(
  "com.vdurmont" % "emoji-java" % "4.0.0"
)

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta"


libraryDependencies ++= Seq(
  "com.twelvemonkeys.common" % "common-lang" % "3.1.2"
)

resolvers += "scalaz-bintray" at "https://dl.bintray.com/scalaz/releases"  
