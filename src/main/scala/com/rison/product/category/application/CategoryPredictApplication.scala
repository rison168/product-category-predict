package com.rison.product.category.application

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, Tokenizer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types.{DataTypes, StringType, StructField, StructType}


/**
 * @author : Rison 2021/9/18 下午3:58
 *         主程序
 */
object CategoryPredictApplication extends Logging{
  /**
   * 特征向量维度
   */
  val numFeatures = 10000
  def main(args: Array[String]): Unit = {
    //TODO: 环境搭建
    val sparkConf: SparkConf = new SparkConf().setMaster("local[*]").setAppName(this.getClass.getSimpleName.stripSuffix("$"))
    val sc: SparkContext = SparkContext.getOrCreate(sparkConf)
    val spark: SparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
     //TODO: 1. 加载数据，（分类标签、原始文件。分词文本数据集合）
    //TODO: 获取数据源
    val trainRDD = sc.textFile("data/train.data")
      .repartition(140)
      .map(
        data => {
          val cateAndName: Array[String] = data.split(" \\|&\\| ")
          Row(cateAndName(0).toDouble, cateAndName(1), cateAndName(2))
        }
      )
    val schema = new StructType(
      Array(
        StructField("label", DataTypes.DoubleType, true),
        StructField("origin", StringType, true),
        StructField("text", StringType, true)
      )
    )
    //TODO: RDD转换为DF
    val originDF: DataFrame = spark.createDataFrame(trainRDD, schema)
    originDF.show(10, false)

    //TODO: 2. 通过贝叶斯分类算法训练商品类目预测模型，特征提取算法为tf-idf
    log.info("start training model with tf-idf......")
    val Array(trainingDF, testingDF): Array[Dataset[Row]] = originDF.randomSplit(Array(0.8, 0.2), 123)

    //TODO: 使用TF-IDF 算法对特征提取 清洗，转换
    val startTime: Long = System.currentTimeMillis()
    val tokenizer: Tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val wordsDF: DataFrame = tokenizer.transform(originDF)
    wordsDF.show(10, false)
    val hashingTF: HashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(numFeatures)
    val featureDF: DataFrame = hashingTF.transform(wordsDF)
    featureDF.show(10, false)
    val idf: IDF = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
    val idfModel: IDFModel = idf.fit(featureDF)
//    idfModel.save("model/idf")
    val rescaledDF: DataFrame = idfModel.transform(featureDF)
    rescaledDF.show(10, false)
//    rescaledDF.cache().count()
//    log.info("Extract feature by tf-idf spends {} ms and train data count is {}", System.currentTimeMillis - startTime, count)

    //TODO: 训练bayes模型
    val naiveBayesModel: NaiveBayesModel = new NaiveBayes()
      .setLabelCol("label")
      .setFeaturesCol("rawFeatures")
      .setPredictionCol("prediction")
      .fit(rescaledDF)
//    naiveBayesModel.save("model/naiveBayes")


    //TODO: 评估模型
    val testWordsDF: DataFrame = tokenizer.transform(testingDF)
    val testFeatureDF: DataFrame = hashingTF.transform(testWordsDF)
    val testRescaleDF: DataFrame = idfModel.transform(testFeatureDF)
    val predictionDF: DataFrame = naiveBayesModel.transform(testRescaleDF)
    predictionDF.select("label", "prediction").show(100, false)
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy: Double = evaluator.evaluate(predictionDF)
    println(accuracy)


    sc.stop()
    spark.stop()

  }
}
