"""
Monte Carlo Simulation for LBO Valuation (Built for Carter's Case Study)
Author: Alex Orta
Purpose: Simulate distribution of IRR and exit multiples under varying assumptions
Enhanced with correlations, defaults, optimization, and multi-tranche debt.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class LBOMonteCarlo:
    """Monte Carlo simulation for LBO returns"""
    
    def __init__(self, n_simulations=10000):
        """
        Initialize the Monte Carlo simulation
        
        Parameters:
        -----------
        n_simulations : int
            Number of Monte Carlo iterations to run
        """
        self.n_simulations = n_simulations
        self.results = None
        
        # Base case assumptions from the model
        self.base_assumptions = {
            'revenue_2001': 537.3,  # $M
            'ebitda_2001': 75.1,    # $M
            'purchase_price': 522.5,  # $M
            'equity_investment': 205,  # $M
            'debt': 317.5,  # $M
            'holding_period': 5,  # years
            'tax_rate': 0.35,
            
            # Growth assumptions
            'revenue_growth_mean': 0.12,  # 12% annual growth
            'revenue_growth_std': 0.03,
            
            # Margin assumptions  
            'ebitda_margin_mean': 0.1398,  # 13.98%
            'ebitda_margin_std': 0.015,
            'gross_margin_mean': 0.413,
            'gross_margin_std': 0.02,
            
            # Capital intensity
            'capex_pct_sales_mean': 0.0382,
            'capex_pct_sales_std': 0.01,
            'nwc_pct_sales_mean': 0.011,
            'nwc_pct_sales_std': 0.005,
            
            # Exit assumptions
            'exit_multiple_mean': 8.13,  # EV/EBITDA
            'exit_multiple_std': 1.2,
            'exit_multiple_min': 6.0,
            'exit_multiple_max': 11.0,
            
            # Debt assumptions
            'debt_paydown_pct': 0.50,  # % of FCF to debt
            'interest_rate_mean': 0.065,
            'interest_rate_std': 0.015,
            
            # New: Debt tranche toggles and params
            'use_senior': True,
            'use_junior': True,
            'use_mezz': True,
            'senior_pct': 0.60,
            'junior_pct': 0.30,
            'mezz_pct': 0.10,
            'senior_rate_mean': 0.05, 'senior_rate_std': 0.01, 'senior_amort_pct': 0.10,
            'junior_rate_mean': 0.08, 'junior_rate_std': 0.015, 'junior_amort_pct': 0.01,
            'mezz_rate_mean': 0.12, 'mezz_rate_std': 0.02, 'mezz_pik': True,
            'leverage_covenant': 5.0  # Max leverage before default
        }
    
    def generate_random_params(self):
        """Generate random parameters for one simulation with correlations"""
        params = {}
        
        # Correlated variables: revenue growth (neg corr with margins), exit multiple (neg corr with interest)
        means = [self.base_assumptions['revenue_growth_mean'], 
                 self.base_assumptions['ebitda_margin_mean'],
                 self.base_assumptions['exit_multiple_mean'],
                 self.base_assumptions['interest_rate_mean']]
        cov_matrix = np.diag([self.base_assumptions['revenue_growth_std']**2,
                              self.base_assumptions['ebitda_margin_std']**2,
                              self.base_assumptions['exit_multiple_std']**2,
                              self.base_assumptions['interest_rate_std']**2])
        # Correlations: -0.5 growth-margin, -0.3 multiple-interest
        cov_matrix[0,1] = cov_matrix[1,0] = -0.5 * self.base_assumptions['revenue_growth_std'] * self.base_assumptions['ebitda_margin_std']
        cov_matrix[2,3] = cov_matrix[3,2] = -0.3 * self.base_assumptions['exit_multiple_std'] * self.base_assumptions['interest_rate_std']
        correlated_samples = np.random.multivariate_normal(means, cov_matrix, self.base_assumptions['holding_period'])
        
        params['revenue_growth'] = np.clip(correlated_samples[:, 0], 0.05, 0.25)
        params['ebitda_margin'] = np.clip(correlated_samples[:, 1], 0.10, 0.20)
        params['exit_multiple'] = np.clip(np.random.triangular(
            self.base_assumptions['exit_multiple_min'],
            correlated_samples[0, 2],  # Use first for scalar
            self.base_assumptions['exit_multiple_max']
        ), self.base_assumptions['exit_multiple_min'], self.base_assumptions['exit_multiple_max'])
        params['interest_rate'] = np.clip(correlated_samples[:, 3], 0.04, 0.12)
        
        # Other params (unchanged)
        params['capex_pct'] = np.clip(
            np.random.normal(
                self.base_assumptions['capex_pct_sales_mean'],
                self.base_assumptions['capex_pct_sales_std'],
                self.base_assumptions['holding_period']
            ),
            0.02, 0.08
        )
        params['nwc_pct'] = np.abs(np.random.normal(
            self.base_assumptions['nwc_pct_sales_mean'],
            self.base_assumptions['nwc_pct_sales_std'],
            self.base_assumptions['holding_period']
        ))
        
        # Tranche-specific rates
        if self.base_assumptions['use_senior']:
            params['senior_rate'] = np.clip(np.random.normal(self.base_assumptions['senior_rate_mean'], self.base_assumptions['senior_rate_std']), 0.04, 0.10)
        if self.base_assumptions['use_junior']:
            params['junior_rate'] = np.clip(np.random.normal(self.base_assumptions['junior_rate_mean'], self.base_assumptions['junior_rate_std']), 0.06, 0.12)
        if self.base_assumptions['use_mezz']:
            params['mezz_rate'] = np.clip(np.random.normal(self.base_assumptions['mezz_rate_mean'], self.base_assumptions['mezz_rate_std']), 0.08, 0.15)
        
        return params
    
    def calculate_lbo_returns(self, params):
        """
        Calculate IRR and MOIC for one set of parameters with multi-tranche debt and defaults
        
        Returns:
        --------
        dict with IRR, MOIC, exit_value, final_debt, equity_value, default_year (if any)
        """
        # Initialize
        revenue = self.base_assumptions['revenue_2001']
        equity_invested = self.base_assumptions['equity_investment']
        holding_period = self.base_assumptions['holding_period']
        tax_rate = self.base_assumptions['tax_rate']
        
        # Multi-tranche debt setup
        total_debt = self.base_assumptions['debt']
        tranches = {}
        if self.base_assumptions['use_senior']:
            tranches['senior'] = {'balance': total_debt * self.base_assumptions['senior_pct'], 
                                  'rate': params.get('senior_rate', self.base_assumptions['senior_rate_mean']),
                                  'amort_pct': self.base_assumptions['senior_amort_pct']}
        if self.base_assumptions['use_junior']:
            tranches['junior'] = {'balance': total_debt * self.base_assumptions['junior_pct'], 
                                  'rate': params.get('junior_rate', self.base_assumptions['junior_rate_mean']),
                                  'amort_pct': self.base_assumptions['junior_amort_pct']}
        if self.base_assumptions['use_mezz']:
            tranches['mezz'] = {'balance': total_debt * self.base_assumptions['mezz_pct'], 
                                'rate': params.get('mezz_rate', self.base_assumptions['mezz_rate_mean']),
                                'amort_pct': 0.0, 'pik': self.base_assumptions['mezz_pik']}
        
        # Redistribute if not all tranches used
        active_pct = sum([self.base_assumptions[f'{name}_pct'] for name in tranches])
        if active_pct < 1.0 and active_pct > 0:
            scale = 1.0 / active_pct
            for t in tranches.values():
                t['balance'] *= scale
        
        cash_flows = []
        
        # Project cash flows year by year
        for year in range(holding_period):
            # Revenue growth
            revenue *= (1 + params['revenue_growth'][year])
            
            # EBITDA
            ebitda = revenue * params['ebitda_margin'][year]
            
            # Simplified D&A (3-4% of revenue)
            depreciation = revenue * 0.035
            
            # EBIT
            ebit = ebitda - depreciation
            
            # Interest expense by tranche
            interest = 0
            for name, t in tranches.items():
                int_exp = t['balance'] * t['rate']
                if name == 'mezz' and t.get('pik', False):
                    t['balance'] += int_exp  # PIK accrues
                else:
                    interest += int_exp
            
            # EBT and taxes
            ebt = ebit - interest
            taxes = max(0, ebt * tax_rate)
            
            # Net income
            net_income = ebt - taxes
            
            # FCF calculation
            capex = revenue * params['capex_pct'][year]
            nwc_increase = revenue * params['revenue_growth'][year] * params['nwc_pct'][year]
            
            fcf = ebitda - capex - nwc_increase - interest - taxes
            
            # Debt paydown: Prioritize senior > junior > mezz
            debt_paydown = 0
            remaining_fcf = max(0, fcf * self.base_assumptions['debt_paydown_pct'])
            for name in ['senior', 'junior', 'mezz']:
                if name in tranches and remaining_fcf > 0:
                    amort = tranches[name]['balance'] * tranches[name]['amort_pct']
                    paydown = min(amort + remaining_fcf, tranches[name]['balance'])
                    tranches[name]['balance'] = max(0, tranches[name]['balance'] - paydown)
                    debt_paydown += paydown
                    remaining_fcf -= (paydown - amort)
            
            # Covenant check: Total leverage
            current_debt = sum(t['balance'] for t in tranches.values())
            if ebitda > 0 and current_debt / ebitda > self.base_assumptions['leverage_covenant']:
                full_cash_flows = [-equity_invested] + cash_flows + [0] * (holding_period - year)
                return {
                    'irr': -1.0,  # Full loss on default
                    'moic': 0.0,
                    'exit_enterprise_value': 0,
                    'exit_equity_value': 0,
                    'final_debt': current_debt,
                    'final_revenue': revenue,
                    'final_ebitda': ebitda,
                    'exit_multiple': params['exit_multiple'],
                    'avg_revenue_growth': np.mean(params['revenue_growth']),
                    'avg_ebitda_margin': np.mean(params['ebitda_margin']),
                    'default_year': year + 1,
                    'default_prob': 1
                }
            
            cash_flows.append(fcf - debt_paydown)
        
        # Exit value calculation
        exit_ebitda = revenue * params['ebitda_margin'][-1]
        exit_enterprise_value = exit_ebitda * params['exit_multiple']
        final_debt = sum(t['balance'] for t in tranches.values())
        exit_equity_value = max(0, exit_enterprise_value - final_debt)
        
        # Calculate returns
        cash_flows_with_exit = cash_flows.copy()
        cash_flows_with_exit[-1] += exit_equity_value
        
        full_cash_flows = [-equity_invested] + cash_flows_with_exit
        try:
            irr = np.irr(full_cash_flows)
        except:
            total_return = exit_equity_value / equity_invested if equity_invested > 0 else 0
            irr = (total_return ** (1/holding_period)) - 1 if holding_period > 0 else 0
        
        moic = exit_equity_value / equity_invested if equity_invested > 0 else 0
        
        return {
            'irr': irr,
            'moic': moic,
            'exit_enterprise_value': exit_enterprise_value,
            'exit_equity_value': exit_equity_value,
            'final_debt': final_debt,
            'final_revenue': revenue,
            'final_ebitda': exit_ebitda,
            'exit_multiple': params['exit_multiple'],
            'avg_revenue_growth': np.mean(params['revenue_growth']),
            'avg_ebitda_margin': np.mean(params['ebitda_margin']),
            'default_year': None,
            'default_prob': 0
        }
    
    def run_simulation(self):
        """Run Monte Carlo simulation"""
        print(f"Running {self.n_simulations:,} Monte Carlo simulations...")
        print("This may take a minute...\n")
        
        results = []
        
        for i in range(self.n_simulations):
            if (i + 1) % 2000 == 0:
                print(f"Completed {i+1:,} simulations...")
            
            params = self.generate_random_params()
            result = self.calculate_lbo_returns(params)
            results.append(result)
        
        self.results = pd.DataFrame(results)
        print(f"\nSimulation complete! Generated {len(self.results):,} scenarios.\n")
        
        return self.results
    
    def analyze_results(self):
        """Generate summary statistics and analysis"""
        if self.results is None:
            raise ValueError("Must run simulation first!")
        
        print("="*70)
        print("MONTE CARLO SIMULATION RESULTS - CARTER'S LBO")
        print("="*70)
        print()
        
        # IRR Analysis
        print("IRR DISTRIBUTION:")
        print("-" * 70)
        print(f"  Mean:              {self.results['irr'].mean():.2%}")
        print(f"  Median:            {self.results['irr'].median():.2%}")
        print(f"  Std Dev:           {self.results['irr'].std():.2%}")
        print(f"  5th Percentile:    {self.results['irr'].quantile(0.05):.2%}")
        print(f"  25th Percentile:   {self.results['irr'].quantile(0.25):.2%}")
        print(f"  75th Percentile:   {self.results['irr'].quantile(0.75):.2%}")
        print(f"  95th Percentile:   {self.results['irr'].quantile(0.95):.2%}")
        print(f"  Min:               {self.results['irr'].min():.2%}")
        print(f"  Max:               {self.results['irr'].max():.2%}")
        print()
        
        # Probability of success
        prob_above_20 = (self.results['irr'] > 0.20).mean()
        prob_above_25 = (self.results['irr'] > 0.25).mean()
        prob_above_30 = (self.results['irr'] > 0.30).mean()
        prob_below_15 = (self.results['irr'] < 0.15).mean()
        prob_default = self.results['default_prob'].mean()
        
        print(f"  Probability IRR > 20%:  {prob_above_20:.1%}")
        print(f"  Probability IRR > 25%:  {prob_above_25:.1%}")
        print(f"  Probability IRR > 30%:  {prob_above_30:.1%}")
        print(f"  Probability IRR < 15%:  {prob_below_15:.1%}")
        print(f"  Probability of Default: {prob_default:.1%}")
        print()
        
        # MOIC Analysis
        print("MOIC DISTRIBUTION:")
        print("-" * 70)
        print(f"  Mean:              {self.results['moic'].mean():.2f}x")
        print(f"  Median:            {self.results['moic'].median():.2f}x")
        print(f"  Std Dev:           {self.results['moic'].std():.2f}x")
        print(f"  25th Percentile:   {self.results['moic'].quantile(0.25):.2f}x")
        print(f"  75th Percentile:   {self.results['moic'].quantile(0.75):.2f}x")
        print()
        
        # Exit Multiple Analysis
        print("EXIT MULTIPLE (EV/EBITDA) DISTRIBUTION:")
        print("-" * 70)
        print(f"  Mean:              {self.results['exit_multiple'].mean():.2f}x")
        print(f"  Median:            {self.results['exit_multiple'].median():.2f}x")
        print(f"  25th Percentile:   {self.results['exit_multiple'].quantile(0.25):.2f}x")
        print(f"  75th Percentile:   {self.results['exit_multiple'].quantile(0.75):.2f}x")
        print()
        
        # Operational metrics
        print("OPERATIONAL METRICS AT EXIT:")
        print("-" * 70)
        print(f"  Final Revenue:     ${self.results['final_revenue'].mean():.1f}M")
        print(f"  Final EBITDA:      ${self.results['final_ebitda'].mean():.1f}M")
        print(f"  Avg Revenue Growth: {self.results['avg_revenue_growth'].mean():.2%}")
        print(f"  Avg EBITDA Margin:  {self.results['avg_ebitda_margin'].mean():.2%}")
        print()
        
        # Exit values
        print("EXIT VALUES:")
        print("-" * 70)
        print(f"  Mean EV:           ${self.results['exit_enterprise_value'].mean():.1f}M")
        print(f"  Mean Equity Value: ${self.results['exit_equity_value'].mean():.1f}M")
        print(f"  Median EV:         ${self.results['exit_enterprise_value'].median():.1f}M")
        print(f"  Median Equity Val: ${self.results['exit_equity_value'].median():.1f}M")
        print()
        
        print("="*70)
        
    def create_visualizations(self, save_path='.'):
        """Create comprehensive visualization dashboard"""
        if self.results is None:
            raise ValueError("Must run simulation first!")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 12)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. IRR Distribution
        ax1 = fig.add_subplot(gs[0, :2])
        self.results['irr'].hist(bins=50, alpha=0.7, color='steelblue', edgecolor='black', ax=ax1)
        ax1.axvline(self.results['irr'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {self.results["irr"].mean():.2%}')
        ax1.axvline(self.results['irr'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {self.results["irr"].median():.2%}')
        ax1.axvline(0.20, color='orange', linestyle=':', linewidth=2, label='20% Target')
        ax1.set_xlabel('IRR', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title("Distribution of IRR Outcomes", fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Format x-axis as percentages
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        
        # 2. IRR Box Plot
        ax2 = fig.add_subplot(gs[0, 2])
        bp = ax2.boxplot(self.results['irr'], vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_edgecolor('darkblue')
        ax2.set_ylabel('IRR', fontsize=12, fontweight='bold')
        ax2.set_title('IRR Range', fontsize=12, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax2.grid(alpha=0.3, axis='y')
        
        # 3. MOIC Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        self.results['moic'].hist(bins=40, alpha=0.7, color='forestgreen', edgecolor='black', ax=ax3)
        ax3.axvline(self.results['moic'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {self.results["moic"].mean():.2f}x')
        ax3.axvline(3.0, color='orange', linestyle=':', linewidth=2, label='3.0x Target')
        ax3.set_xlabel('MOIC', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('MOIC Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        
        # 4. Exit Multiple Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        self.results['exit_multiple'].hist(bins=40, alpha=0.7, color='coral', edgecolor='black', ax=ax4)
        ax4.axvline(self.results['exit_multiple'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {self.results["exit_multiple"].mean():.2f}x')
        ax4.set_xlabel('Exit Multiple (EV/EBITDA)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title('Exit Multiple Distribution', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        # 5. IRR vs Exit Multiple Scatter
        ax5 = fig.add_subplot(gs[1, 2])
        scatter = ax5.scatter(self.results['exit_multiple'], self.results['irr'], 
                              alpha=0.3, c=self.results['moic'], cmap='viridis', s=20)
        ax5.set_xlabel('Exit Multiple', fontsize=12, fontweight='bold')
        ax5.set_ylabel('IRR', fontsize=12, fontweight='bold')
        ax5.set_title('IRR vs Exit Multiple', fontsize=12, fontweight='bold')
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        plt.colorbar(scatter, ax=ax5, label='MOIC')
        ax5.grid(alpha=0.3)
        
        # 6. Revenue Growth vs IRR
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.scatter(self.results['avg_revenue_growth'], self.results['irr'], 
                    alpha=0.3, color='purple', s=20)
        ax6.set_xlabel('Avg Revenue Growth', fontsize=12, fontweight='bold')
        ax6.set_ylabel('IRR', fontsize=12, fontweight='bold')
        ax6.set_title('Revenue Growth Impact on IRR', fontsize=12, fontweight='bold')
        ax6.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax6.grid(alpha=0.3)
        
        # 7. EBITDA Margin vs IRR
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.scatter(self.results['avg_ebitda_margin'], self.results['irr'], 
                    alpha=0.3, color='teal', s=20)
        ax7.set_xlabel('Avg EBITDA Margin', fontsize=12, fontweight='bold')
        ax7.set_ylabel('IRR', fontsize=12, fontweight='bold')
        ax7.set_title('EBITDA Margin Impact on IRR', fontsize=12, fontweight='bold')
        ax7.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax7.grid(alpha=0.3)
        
        # 8. Cumulative Probability Distribution
        ax8 = fig.add_subplot(gs[2, 2])
        sorted_irr = np.sort(self.results['irr'])
        cumulative = np.arange(1, len(sorted_irr) + 1) / len(sorted_irr)
        ax8.plot(sorted_irr, cumulative, linewidth=2, color='darkblue')
        ax8.axvline(0.20, color='red', linestyle='--', linewidth=2, label='20% Target')
        ax8.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax8.set_xlabel('IRR', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax8.set_title('IRR Cumulative Distribution', fontsize=12, fontweight='bold')
        ax8.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax8.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax8.legend(fontsize=9)
        ax8.grid(alpha=0.3)
        
        plt.suptitle("Carter's LBO - Monte Carlo Simulation Results", 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'{save_path}/lbo_monte_carlo_results.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}/lbo_monte_carlo_results.png")
        
        return fig
    
    def sensitivity_analysis(self):
        """Perform sensitivity analysis on key drivers"""
        if self.results is None:
            raise ValueError("Must run simulation first!")
        
        print("\nSENSITIVITY ANALYSIS:")
        print("="*70)
        
        # Calculate correlations
        correlations = self.results[[
            'irr', 'exit_multiple', 'avg_revenue_growth', 
            'avg_ebitda_margin', 'final_revenue'
        ]].corr()['irr'].sort_values(ascending=False)
        
        print("\nCorrelation with IRR:")
        print("-" * 70)
        for idx, val in correlations.items():
            if idx != 'irr':
                print(f"  {idx:25s}: {val:+.3f}")
        
        # Tornado chart data
        print("\n\nKEY VALUE DRIVERS (Tornado Analysis):")
        print("-" * 70)
        
        base_irr = self.results['irr'].median()
        
        # Test each variable at 10th and 90th percentile
        variables = {
            'Exit Multiple': 'exit_multiple',
            'Revenue Growth': 'avg_revenue_growth',
            'EBITDA Margin': 'avg_ebitda_margin'
        }
        
        tornado_data = []
        
        for var_name, var_col in variables.items():
            p10 = self.results[var_col].quantile(0.10)
            p90 = self.results[var_col].quantile(0.90)
            
            irr_p10 = self.results[self.results[var_col] <= p10]['irr'].median()
            irr_p90 = self.results[self.results[var_col] >= p90]['irr'].median()
            
            impact = abs(irr_p90 - irr_p10)
            
            tornado_data.append({
                'Variable': var_name,
                'P10 Value': p10,
                'P90 Value': p90,
                'IRR at P10': irr_p10,
                'IRR at P90': irr_p90,
                'IRR Range': impact
            })
            
            print(f"\n{var_name}:")
            print(f"  P10 = {p10:.3f} → IRR = {irr_p10:.2%}")
            print(f"  P90 = {p90:.3f} → IRR = {irr_p90:.2%}")
            print(f"  Impact on IRR: {impact:.2%}")
        
        return pd.DataFrame(tornado_data)
    
    def export_results(self, filename='monte_carlo_results.csv'):
        """Export detailed results to CSV"""
        if self.results is None:
            raise ValueError("Must run simulation first!")
        
        self.results.to_csv(filename, index=False)
        print(f"\nResults exported to: {filename}")
        
        # Also create summary statistics file
        summary = self.results.describe().T
        summary.to_csv(filename.replace('.csv', '_summary.csv'))
        print(f"Summary statistics exported to: {filename.replace('.csv', '_summary.csv')}")
    
    def optimize_debt(self, target_irr=0.25):
        """Optimize debt percentage for target IRR using single-run approximation"""
        def objective(debt_pct):
            # Temporarily update assumptions
            original_debt = self.base_assumptions['debt']
            self.base_assumptions['debt'] = self.base_assumptions['purchase_price'] * debt_pct
            self.base_assumptions['equity_investment'] = self.base_assumptions['purchase_price'] - self.base_assumptions['debt']
            
            # Run a single simulation for approximation
            params = self.generate_random_params()
            result = self.calculate_lbo_returns(params)
            
            # Restore
            self.base_assumptions['debt'] = original_debt
            self.base_assumptions['equity_investment'] = self.base_assumptions['purchase_price'] - original_debt
            
            return abs(result['irr'] - target_irr)
        
        res = minimize_scalar(objective, bounds=(0.4, 0.8), method='bounded')  # 40-80% debt range
        optimal_debt_pct = res.x
        print(f"Optimal Debt %: {optimal_debt_pct:.2%} for target IRR of {target_irr:.0%}")
        return optimal_debt_pct


def main():
    """Main execution function"""
    print("="*70)
    print("CARTER'S LBO - MONTE CARLO SIMULATION")
    print("="*70)
    print()
    print("This simulation models the distribution of returns for the Carter's")
    print("leveraged buyout under varying operational and market assumptions.")
    print()
    
    # Initialize simulation
    mc = LBOMonteCarlo(n_simulations=10000)
    
    # Run simulation
    results = mc.run_simulation()
    
    # Analyze results
    mc.analyze_results()
    
    # Sensitivity analysis
    tornado = mc.sensitivity_analysis()
    
    # Create visualizations
    mc.create_visualizations()
    
    # Export results
    mc.export_results()
    
    # Run optimization
    mc.optimize_debt(target_irr=0.25)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print(f"  • Median IRR: {results['irr'].median():.2%}")
    print(f"  • Probability of >20% IRR: {(results['irr'] > 0.20).mean():.1%}")
    print(f"  • Expected MOIC: {results['moic'].median():.2f}x")
    print(f"  • Default Probability: {results['default_prob'].mean():.1%}")
    print("="*70)


if __name__ == "__main__":
    main()