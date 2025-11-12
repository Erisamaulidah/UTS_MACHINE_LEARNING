using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace HoaxNewsDetectionML
{
    public class NewsData
    {
        [LoadColumn(1)] public string? Title { get; set; }
        [LoadColumn(2)] public string? Text { get; set; }
        [LoadColumn(3)] public string? Label { get; set; }
    }

    public class NewsPrediction
    {
        [ColumnName("PredictedLabel")]
        public string? PredictedLabel { get; set; }
    }

    class Program
    {
        static void Main()
        {
            var mlContext = new MLContext(seed: 0);
            string dataPath = "news_clean.csv";

            Console.WriteLine("📂 Loading dataset...");
            var data = mlContext.Data.LoadFromTextFile<NewsData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = mlContext.Transforms.Conversion
                .MapValueToKey("Label", "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText("TitleFeats", "Title"))
                .Append(mlContext.Transforms.Text.FeaturizeText("TextFeats", "Text"))
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeats", "TextFeats"))
                .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(
                    mlContext.BinaryClassification.Trainers.FastTree()))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("🚀 Training model...");
            var model = pipeline.Fit(split.TrainSet);

            Console.WriteLine("📊 Evaluating model...");
            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"\n=== Model Evaluation ===");
            Console.WriteLine($"Macro Accuracy : {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"Micro Accuracy : {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"Log Loss       : {metrics.LogLoss:F4}");

            mlContext.Model.Save(model, split.TrainSet.Schema, "HoaxNewsModel.zip");
            Console.WriteLine("\n💾 Model saved successfully! -> HoaxNewsModel.zip");

            var predEngine = mlContext.Model.CreatePredictionEngine<NewsData, NewsPrediction>(model);
            var sample = new NewsData
            {
                Title = "Pemerintah akan membagikan uang 10 juta kepada semua warga",
                Text = "Berita ini viral di media sosial tapi tidak ada sumber resmi dari pemerintah."
            };

            var result = predEngine.Predict(sample);
            Console.WriteLine($"\n📰 Prediksi contoh:");
            Console.WriteLine($"Judul: {sample.Title}");
            Console.WriteLine($"Prediksi Label: {result.PredictedLabel}");
        }
    }
}
