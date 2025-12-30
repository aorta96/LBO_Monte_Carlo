"""
Advanced LBO Monte Carlo Analysis
Wraps LBOMonteCarlo with scenario analysis, stress testing,
sensitivity, and what-if exploration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from lbo_monte_carlo import LBOMonteCarlo


class AdvancedLBOAnalysis:
    """
    Advanced analysis layer that wraps LBOMonteCarlo.
    No simulation logic is reimplemented here.
    """

    REQUIRED_COLUMNS = {
        "irr",
        "moic",
        "exit_multiple",
        "avg_revenue_growth",
        "avg_ebitda_margin",
    }

    def __init__(self, lbo_model: LBOMonteCarlo):
        self.lbo = lbo_model
        self.base_results = None

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------
    def run_base_simulation(self):
        print("\nRunning base Monte Carlo simulation...")
        self.lbo.run_simulation()
        self.base_results = self.lbo.results.copy()
        self._validate_results(self.base_results)

    def _validate_results(self, df: pd.DataFrame):
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"LBOMonteCarlo output missing columns: {missing}")

    # ------------------------------------------------------------------
    # Scenario Analysis
    # ------------------------------------------------------------------
    def scenario_analysis(self, scenarios=None, n_simulations=1000):
        if scenarios is None:
            scenarios = {
                "Bear Case": {
                    "revenue_growth_mean": 0.08,
                    "ebitda_margin_mean": 0.125,
                    "exit_multiple_mean": 7.0,
                },
                "Base Case": {
                    "revenue_growth_mean": 0.12,
                    "ebitda_margin_mean": 0.1398,
                    "exit_multiple_mean": 8.13,
                },
                "Bull Case": {
                    "revenue_growth_mean": 0.16,
                    "ebitda_margin_mean": 0.155,
                    "exit_multiple_mean": 9.5,
                },
            }

        print("\n" + "=" * 70)
        print("SCENARIO ANALYSIS")
        print("=" * 70)

        results = {}

        for name, overrides in scenarios.items():
            df = self.lbo.run_with_assumptions(
                overrides=overrides,
                n_simulations=n_simulations,
            )
            self._validate_results(df)
            results[name] = df

            print(f"\n{name}:")
            print(f"  Median IRR:   {df['irr'].median():.2%}")
            print(f"  Mean MOIC:    {df['moic'].mean():.2f}x")
            print(f"  5th Pct IRR:  {df['irr'].quantile(0.05):.2%}")
            print(f"  95th Pct IRR: {df['irr'].quantile(0.95):.2%}")

        self._plot_scenario_comparison(results)
        return results

    # ------------------------------------------------------------------
    # Stress Testing
    # ------------------------------------------------------------------
    def stress_test(self, n_simulations=500):
        print("\n" + "=" * 70)
        print("STRESS TESTING")
        print("=" * 70)

        stress_scenarios = {
            "Recession": {
                "revenue_growth_mean": 0.02,
                "ebitda_margin_mean": 0.11,
                "exit_multiple_mean": 5.5,
            },
            "Margin Compression": {
                "ebitda_margin_mean": 0.10,
            },
            "Multiple Contraction": {
                "exit_multiple_mean": 6.0,
            },
            'High Interest Rates': {
                'senior_rate_mean': 0.08,
                'junior_rate_mean': 0.11,
                'mezz_rate_mean': 0.15
            }

        }

        results = {}

        for name, overrides in stress_scenarios.items():
            df = self.lbo.run_with_assumptions(
                overrides=overrides,
                n_simulations=n_simulations,
            )
            self._validate_results(df)
            results[name] = df

            prob_loss = (df["irr"] < 0).mean()
            prob_below_hurdle = (df["irr"] < 0.15).mean()

            print(f"\n{name}:")
            print(f"  Median IRR:     {df['irr'].median():.2%}")
            print(f"  5th Pct IRR:    {df['irr'].quantile(0.05):.2%}")
            print(f"  Prob of Loss:   {prob_loss:.1%}")
            print(f"  IRR < 15%:      {prob_below_hurdle:.1%}")
            print(f"  Median MOIC:    {df['moic'].median():.2f}x")

        return results

    # ------------------------------------------------------------------
    # Sensitivity (Tornado)
    # ------------------------------------------------------------------
    def sensitivity_tornado_chart(self):
        if self.base_results is None:
            raise ValueError("Run base simulation first.")

        print("\n" + "=" * 70)
        print("CREATING TORNADO CHART")
        print("=" * 70)

        variables = {
            "Exit Multiple": "exit_multiple",
            "Revenue Growth": "avg_revenue_growth",
            "EBITDA Margin": "avg_ebitda_margin",
        }

        base_irr = self.base_results["irr"].median()
        rows = []

        for label, col in variables.items():
            p10 = self.base_results[col].quantile(0.10)
            p90 = self.base_results[col].quantile(0.90)

            irr_low = self.base_results[self.base_results[col] <= p10]["irr"].median()
            irr_high = self.base_results[self.base_results[col] >= p90]["irr"].median()

            rows.append({
                "Variable": label,
                "Low": irr_low - base_irr,
                "High": irr_high - base_irr,
                "Range": abs(irr_high - irr_low),
            })

        df = pd.DataFrame(rows).sort_values("Range")

        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(df))

        ax.barh(y, df["Low"], color="#d62728", alpha=0.7, label="Downside")
        ax.barh(y, df["High"], color="#2ca02c", alpha=0.7, label="Upside")
        ax.axvline(0, color="black", linewidth=2)

        ax.set_yticks(y)
        ax.set_yticklabels(df["Variable"])
        ax.set_xlabel("Impact on IRR")
        ax.set_title("Tornado Chart: Sensitivity to Key Variables")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0%}"))
        ax.legend()
        ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        filename = "tornado_chart.png"
        plt.savefig(filename, dpi=300)
        print(f"Tornado chart saved to: {os.path.abspath(filename)}")

        return df

    # ------------------------------------------------------------------
    # What-if Analysis
    # ------------------------------------------------------------------
    def what_if_analysis(self, parameter, values, n_simulations=500):
        print("\n" + "=" * 70)
        print(f"WHAT-IF ANALYSIS: {parameter}")
        print("=" * 70)

        rows = []

        for value in values:
            df = self.lbo.run_with_assumptions(
                overrides={parameter: value},
                n_simulations=n_simulations,
            )
            self._validate_results(df)

            rows.append({
                "Parameter Value": value,
                "Median IRR": df["irr"].median(),
                "Mean IRR": df["irr"].mean(),
                "P25 IRR": df["irr"].quantile(0.25),
                "P75 IRR": df["irr"].quantile(0.75),
                "Median MOIC": df["moic"].median(),
            })

            print(f"  {parameter} = {value:.3f} → Median IRR = {df['irr'].median():.2%}")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def _plot_scenario_comparison(self, scenario_results):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ["#d62728", "#1f77b4", "#2ca02c"]

        for i, (name, df) in enumerate(scenario_results.items()):
            axes[0].hist(df["irr"], bins=30, alpha=0.6, color=colors[i], label=name)

        axes[0].set_title("IRR Distribution by Scenario")
        axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].boxplot(
            [df["irr"] for df in scenario_results.values()],
            labels=scenario_results.keys(),
            patch_artist=True,
        )
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        axes[1].set_title("IRR Range by Scenario")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        filename = "scenario_comparison.png"
        plt.savefig(filename, dpi=300)
        print(f"Scenario comparison saved to: {os.path.abspath(filename)}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("ADVANCED LBO MONTE CARLO ANALYSIS")
    print("=" * 70)

    lbo = LBOMonteCarlo(n_simulations=10_000)
    advanced = AdvancedLBOAnalysis(lbo)

    advanced.run_base_simulation()
    advanced.scenario_analysis()
    advanced.stress_test()
    advanced.sensitivity_tornado_chart()

    advanced.what_if_analysis(
        "exit_multiple_mean", [6, 7, 8, 9, 10, 11]
    )

    advanced.what_if_analysis(
        "revenue_growth_mean", [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    )

    print("\n" + "=" * 70)
    print("ADVANCED ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
