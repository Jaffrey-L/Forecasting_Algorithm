import datetime
import json
import unittest

import pandas as pd

from generate_accuracy_report import expand_sku_accuracy_rows


class TestGenerateAccuracyReport(unittest.TestCase):
    def test_expand_rows_basic(self):
        run_date = datetime.date(2026, 3, 10)
        spu_rows = pd.DataFrame(
            [
                {
                    "spu": "2141",
                    "run_date": run_date,
                    "winner_algo": "Ensemble_Wgt",
                    "validation_wmape": 0.12,
                    "sku_accuracy_json": json.dumps(
                        {
                            "SKU_A": {"wmape": 0.10, "total_sales": 1000, "weight_in_spu": 0.8},
                            "SKU_B": {"wmape": 0.20, "total_sales": 250, "weight_in_spu": 0.2},
                        },
                        ensure_ascii=False,
                    ),
                }
            ]
        )

        detail_df, summary_df = expand_sku_accuracy_rows(spu_rows)
        self.assertEqual(len(detail_df), 2)
        self.assertEqual(len(summary_df), 1)

        weighted = summary_df.iloc[0]["sku_weighted_wmape"]
        self.assertAlmostEqual(weighted, 0.12, places=6)

    def test_expand_rows_error(self):
        run_date = datetime.date(2026, 3, 10)
        spu_rows = pd.DataFrame(
            [
                {
                    "spu": "2141",
                    "run_date": run_date,
                    "winner_algo": "Ensemble_Wgt",
                    "validation_wmape": 0.12,
                    "sku_accuracy_json": json.dumps({"error": "failed"}, ensure_ascii=False),
                }
            ]
        )

        detail_df, summary_df = expand_sku_accuracy_rows(spu_rows)
        self.assertEqual(len(detail_df), 0)
        self.assertEqual(len(summary_df), 1)
        self.assertEqual(summary_df.iloc[0]["error"], "failed")


if __name__ == "__main__":
    unittest.main()

