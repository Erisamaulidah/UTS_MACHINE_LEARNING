// File: NewsData.cs
using Microsoft.ML.Data;

namespace FakeNewsDetection
{
    // Class untuk membaca data dari CSV
    public class NewsDataInput
    {
        [LoadColumn(0)] public string? title { get; set; }
        [LoadColumn(1)] public string? text { get; set; }
      [LoadColumn(2)] public string? label { get; set; }   
    }

    // Class untuk proses model (dengan Label)
    public class NewsData
    {
        public string? title { get; set; }
        public bool? text { get; set; }
        public bool? label { get; set; }
    }

    // Class untuk prediksi
    public class NewsPrediction
    {
        [ColumnName("PredictedLabel")] public bool IsFake { get; set; }
        public float Score { get; set; }
    }
}