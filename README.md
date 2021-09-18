# 基于机器学习的商品类目预测

## 项目介绍

项目根据商品名称把商品预测出商品所属的类目。本项目中商品类目分为三级，一共有 962 个三级类目，完整的类目文件见：[src/main/resources/category.json](src/main/resources/category.json) 。比如名称为“荣耀手机原装华为p9p10plus/mate10/9/8/nova2s/3e荣耀9i/v10手机耳机【线控带麦】AM115标配版白色”的商品，该商品的实际类目是【手机耳机】，我们需要训练一个模型能够根据商品名称自动预测出该商品属于【手机耳机】类目。

项目基于 JAVA + Scala语言开发，使用 Spring Boot 开发框架和 Spark MLlib 机器学习框架，以 RESTful 接口的方式对外提供服务。
