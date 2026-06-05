1. Concept labels

原始 DREAM 使用分析師或 malware behavior report 整理出的行為概念標籤，而專案目前是從 KronoDroid 的 permission/static feature columns 推導出概念標籤。

2. Explainer usage

原始 DREAM 會將修正後的 explanations 用於 adaptation，而專案目前的 explainer 主要是產生並印出 explanation masks，還沒有把這些 masks 回饋進訓練流程。

3. Human feedback

原始 DREAM 假設有分析師提供 malware family label 和 concept revision，而專案目前用資料集中的 labels 與自動推導出的 concepts 來模擬這個 feedback。

4. Classifier coverage

原始 DREAM 評估多種 classifier，例如 Drebin、Mamadroid 和 Damd，而專案目前只實作一個 Drebin-style MLP classifier。

5. Holdout evaluation

原始 DREAM 會對多個 held-out families 做實驗並取平均結果，而專案目前一次只測試一個 held-out family。

6. Baseline comparison

原始 DREAM 會和多個 drift detection / adaptation baselines 比較，而專案目前主要只呈現 DREAM-style 方法本身的結果。

7. Contrastive loss

原始 DREAM 使用官方實作中的 contrastive loss，而專案目前使用 CADE-style contrastive separation loss 作為近似版本。

8. Learning-rate schedule

原始 DREAM 的 adaptation schedule 可能包含更多官方實作細節，而專案目前使用較簡化的 threshold-based detector learning-rate reduction。
