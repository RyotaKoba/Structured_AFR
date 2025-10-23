import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# PDF保存用の設定
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')  # GUIなしでの実行用

# 日本語フォント設定（サーバ環境での代替）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

class AFRAnalyzer:
    def __init__(self, data_dir, output_dir="analysis_results", save_format='both'):
        """
        AFRスコア分析クラス
        
        Args:
            data_dir: 保存されたスコアデータのディレクトリ
            output_dir: 分析結果（画像等）の保存先
            save_format: 'png', 'pdf', 'both' のいずれか
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.save_format = save_format
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading data from {data_dir}...")
        self.fo_scores, self.snip_scores, self.metadata = self.load_weight_scores()
        self.n_layers = len(self.fo_scores)
        
        print(f"Data loaded successfully!")
        print(f"Number of layers: {self.n_layers}")
        print(f"Score shape per layer: {self.fo_scores[0]['shape']}")
        print(f"Output directory: {output_dir}")
        print(f"Save format: {save_format}")
        
    def save_figure(self, fig, filename):
        """図を指定された形式で保存"""
        base_name = filename.replace('.png', '').replace('.pdf', '')
        
        if self.save_format in ['png', 'both']:
            fig.savefig(os.path.join(self.output_dir, f'{base_name}.png'))
        
        if self.save_format in ['pdf', 'both']:
            fig.savefig(os.path.join(self.output_dir, f'{base_name}.pdf'))
            
    def load_weight_scores(self):
        """保存された重みスコアを読み込む"""
        fo_scores = torch.load(os.path.join(self.data_dir, 'fo_weight_scores.pt'))
        snip_scores = torch.load(os.path.join(self.data_dir, 'snip_weight_scores.pt'))
        
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        return fo_scores, snip_scores, metadata
    
    def get_layer_scores(self, layer_idx):
        """指定層のFOとSNIPスコアを取得"""
        fo_tensor = self.fo_scores[layer_idx]['W_metric']
        snip_tensor = self.snip_scores[layer_idx]['W_metric']
        """トリム平均を使用してニューロンスコアを計算"""
        # 上位5%と下位5%を除外する設定
        trim_percent = 2
        
        # 各列をソートして上位・下位を除外
        sorted_fo, _ = torch.sort(fo_tensor, dim=0)
        sorted_snip, _ = torch.sort(snip_tensor, dim=0)
        n_rows = fo_tensor.shape[0]  # 4096

        # 除外する要素数を計算
        trim_count = int(n_rows * trim_percent / 100) if trim_percent > 0 else 0
        
        # 中央部分を抽出して平均を計算
        if trim_count != 0:
            trimmed_fo = sorted_fo[trim_count:-trim_count, :]
            trimmed_snip = sorted_snip[trim_count:-trim_count, :]
        else:
            trimmed_fo = sorted_fo
            trimmed_snip = sorted_snip
        return trimmed_fo, trimmed_snip
    
    def basic_statistics(self):
        """基本統計量の計算と保存"""
        print("Computing basic statistics...")
        
        results = {
            'layer_stats': [],
            'global_stats': {}
        }
        
        all_fo_values = []
        all_snip_values = []
        
        for i in range(self.n_layers):
            fo_tensor, snip_tensor = self.get_layer_scores(i)
            
            # 各層の統計
            layer_stat = {
                'layer': i,
                'fo_mean': float(fo_tensor.mean()),
                'fo_std': float(fo_tensor.std()),
                'fo_min': float(fo_tensor.min()),
                'fo_max': float(fo_tensor.max()),
                'fo_median': float(fo_tensor.median()),
                'snip_mean': float(snip_tensor.mean()),
                'snip_std': float(snip_tensor.std()),
                'snip_min': float(snip_tensor.min()),
                'snip_max': float(snip_tensor.max()),
                'snip_median': float(snip_tensor.median()),
            }
            results['layer_stats'].append(layer_stat)
            
            # 全体統計用のデータ収集
            all_fo_values.extend(fo_tensor.flatten().tolist())
            all_snip_values.extend(snip_tensor.flatten().tolist())
        
        # 全体統計
        all_fo = torch.tensor(all_fo_values)
        all_snip = torch.tensor(all_snip_values)
        
        results['global_stats'] = {
            'fo_global_mean': float(all_fo.mean()),
            'fo_global_std': float(all_fo.std()),
            'fo_global_range': float(all_fo.max() - all_fo.min()),
            'snip_global_mean': float(all_snip.mean()),
            'snip_global_std': float(all_snip.std()),
            'snip_global_range': float(all_snip.max() - all_snip.min()),
        }
        
        # 結果をJSONで保存
        with open(os.path.join(self.output_dir, 'basic_statistics.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # 統計表の可視化
        self._plot_layer_statistics(results['layer_stats'])
        
        print(f"Basic statistics saved to {self.output_dir}")
        return results
    
    def _plot_layer_statistics(self, layer_stats):
        """層別統計のグラフ化（デュアル軸対応）"""
        df = pd.DataFrame(layer_stats)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Layer-wise Statistics Comparison (FO vs SNIP)', fontsize=16)
        
        # 平均値比較
        ax1 = axes[0, 0]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(df['layer'], df['fo_mean'], 'b-o', label='FO', alpha=0.7)
        line2 = ax2.plot(df['layer'], df['snip_mean'], 'r-s', label='SNIP', alpha=0.7)
        
        ax1.set_title('Mean Values')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('FO Mean', color='blue')
        ax2.set_ylabel('SNIP Mean', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        
        # 凡例を統合
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 標準偏差比較
        ax1 = axes[0, 1]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(df['layer'], df['fo_std'], 'b-o', label='FO', alpha=0.7)
        line2 = ax2.plot(df['layer'], df['snip_std'], 'r-s', label='SNIP', alpha=0.7)
        
        ax1.set_title('Standard Deviation')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('FO Std Dev', color='blue')
        ax2.set_ylabel('SNIP Std Dev', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 範囲比較
        fo_range = df['fo_max'] - df['fo_min']
        snip_range = df['snip_max'] - df['snip_min']
        
        ax1 = axes[0, 2]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(df['layer'], fo_range, 'b-o', label='FO', alpha=0.7)
        line2 = ax2.plot(df['layer'], snip_range, 'r-s', label='SNIP', alpha=0.7)
        
        ax1.set_title('Value Range (Max - Min)')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('FO Range', color='blue')
        ax2.set_ylabel('SNIP Range', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 中央値比較
        ax1 = axes[1, 0]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(df['layer'], df['fo_median'], 'b-o', label='FO', alpha=0.7)
        line2 = ax2.plot(df['layer'], df['snip_median'], 'r-s', label='SNIP', alpha=0.7)
        
        ax1.set_title('Median Values')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('FO Median', color='blue')
        ax2.set_ylabel('SNIP Median', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 平均と標準偏差の関係（散布図なのでデュアル軸不要）
        axes[1, 1].scatter(df['fo_mean'], df['fo_std'], c='blue', alpha=0.7, label='FO', s=50)
        axes[1, 1].scatter(df['snip_mean'], df['snip_std'], c='red', alpha=0.7, label='SNIP', s=50)
        axes[1, 1].set_title('Mean vs Std Deviation')
        axes[1, 1].set_xlabel('Mean')
        axes[1, 1].set_ylabel('Std Dev')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 変動係数（CV）
        fo_cv = df['fo_std'] / (df['fo_mean'].abs() + 1e-8)
        snip_cv = df['snip_std'] / (df['snip_mean'].abs() + 1e-8)
        
        ax1 = axes[1, 2]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(df['layer'], fo_cv, 'b-o', label='FO', alpha=0.7)
        line2 = ax2.plot(df['layer'], snip_cv, 'r-s', label='SNIP', alpha=0.7)
        
        ax1.set_title('Coefficient of Variation (CV)')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('FO CV (Std/Mean)', color='blue')
        ax2.set_ylabel('SNIP CV (Std/Mean)', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        # 修正: save_figureメソッドを使用
        self.save_figure(fig, 'layer_statistics')
        plt.close()
    
    def correlation_analysis(self):
        """FOとSNIPの相関分析"""
        print("Computing correlation analysis...")
        
        correlations = {
            'layer_correlations': [],
            'aggregated_correlations': {}
        }
        
        all_fo_neuron_scores = []
        all_snip_neuron_scores = []
        
        for i in range(self.n_layers):
            fo_tensor, snip_tensor = self.get_layer_scores(i)
            
            # ニューロン単位で集約（元コードと同じ方法）
            fo_neuron = self.calculate_neuron_score_v2(fo_tensor)
            snip_neuron = self.calculate_neuron_score_v2(snip_tensor)
            
            # 相関計算
            fo_flat = fo_neuron.flatten().numpy()
            snip_flat = snip_neuron.flatten().numpy()
            
            pearson_corr, pearson_p = pearsonr(fo_flat, snip_flat)
            spearman_corr, spearman_p = spearmanr(fo_flat, snip_flat)
            
            # コサイン類似度
            cosine_sim = 1 - cosine(fo_flat, snip_flat)
            
            layer_corr = {
                'layer': i,
                'pearson_correlation': float(pearson_corr),
                'pearson_p_value': float(pearson_p),
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'cosine_similarity': float(cosine_sim)
            }
            correlations['layer_correlations'].append(layer_corr)
            
            # 全体集約用のデータ
            all_fo_neuron_scores.extend(fo_flat.tolist())
            all_snip_neuron_scores.extend(snip_flat.tolist())
        
        # 全ニューロンでの集約相関
        all_fo = np.array(all_fo_neuron_scores)
        all_snip = np.array(all_snip_neuron_scores)
        
        global_pearson, global_pearson_p = pearsonr(all_fo, all_snip)
        global_spearman, global_spearman_p = spearmanr(all_fo, all_snip)
        global_cosine = 1 - cosine(all_fo, all_snip)
        
        correlations['aggregated_correlations'] = {
            'global_pearson': float(global_pearson),
            'global_pearson_p': float(global_pearson_p),
            'global_spearman': float(global_spearman),
            'global_spearman_p': float(global_spearman_p),
            'global_cosine_similarity': float(global_cosine)
        }
        
        # 結果保存
        with open(os.path.join(self.output_dir, 'correlation_analysis.json'), 'w') as f:
            json.dump(correlations, f, indent=2)
        
        # 可視化
        self._plot_correlations(correlations, all_fo, all_snip)
        
        print(f"Correlation analysis saved to {self.output_dir}")
        return correlations
    
    def calculate_neuron_score_v2(self, W_metric):
        """元コードのcalculate_neuron_score_v2を再実装"""
        mean_scores = W_metric.mean(axis=0)
        std_scores = W_metric.std(axis=0)
        snr_scores = torch.abs(mean_scores) / (std_scores + 1e-8)


        mean_scores = torch.abs(mean_scores)
        return mean_scores
        # return snr_scores
        
    def _plot_correlations(self, correlations, all_fo, all_snip):
        """相関関係の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FO vs SNIP Correlation Analysis', fontsize=16)
        
        # 層別相関の推移
        df_corr = pd.DataFrame(correlations['layer_correlations'])
        
        axes[0, 0].plot(df_corr['layer'], df_corr['pearson_correlation'], 'b-o', label='Pearson', alpha=0.7)
        axes[0, 0].plot(df_corr['layer'], df_corr['spearman_correlation'], 'r-s', label='Spearman', alpha=0.7)
        axes[0, 0].plot(df_corr['layer'], df_corr['cosine_similarity'], 'g-^', label='Cosine', alpha=0.7)
        axes[0, 0].set_title('Layer-wise Correlations')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Correlation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(-1, 1)
        
        # 全体散布図
        sample_indices = np.random.choice(len(all_fo), min(10000, len(all_fo)), replace=False)
        fo_sample = all_fo[sample_indices]
        snip_sample = all_snip[sample_indices]
        
        axes[0, 1].scatter(fo_sample, snip_sample, alpha=0.5, s=1)
        axes[0, 1].set_title(f'Global Scatter Plot (Pearson: {correlations["aggregated_correlations"]["global_pearson"]:.3f})')
        axes[0, 1].set_xlabel('FO Neuron Scores')
        axes[0, 1].set_ylabel('SNIP Neuron Scores')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 相関ヒートマップ（層別）
        corr_matrix = np.array([df_corr['pearson_correlation'].values, 
                               df_corr['spearman_correlation'].values,
                               df_corr['cosine_similarity'].values])
        
        im = axes[1, 0].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_title('Correlation Heatmap by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_yticks([0, 1, 2])
        axes[1, 0].set_yticklabels(['Pearson', 'Spearman', 'Cosine'])
        
        # カラーバー
        cbar = plt.colorbar(im, ax=axes[1, 0])
        cbar.set_label('Correlation')
        
        # 分布比較（対数スケール）
        axes[1, 1].hist(np.log10(np.abs(all_fo) + 1e-10), bins=50, alpha=0.5, label='FO (log10)', density=True)
        axes[1, 1].hist(np.log10(np.abs(all_snip) + 1e-10), bins=50, alpha=0.5, label='SNIP (log10)', density=True)
        axes[1, 1].set_title('Score Distribution Comparison (Log Scale)')
        axes[1, 1].set_xlabel('Log10(|Score|)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # 修正: save_figureメソッドを使用
        self.save_figure(fig, 'correlation_analysis')
        plt.close()    
    
    def _create_layer_statistics_figure(self, layer_stats):
        """層別統計図を生成（PDF用）"""
        df = pd.DataFrame(layer_stats)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Layer-wise Statistics Comparison (FO vs SNIP)', fontsize=16)
        
        # ... (前述のデュアル軸コードと同じ)
        # 平均値比較
        ax1 = axes[0, 0]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(df['layer'], df['fo_mean'], 'b-o', label='FO', alpha=0.7)
        line2 = ax2.plot(df['layer'], df['snip_mean'], 'r-s', label='SNIP', alpha=0.7)
        
        ax1.set_title('Mean Values')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('FO Mean', color='blue')
        ax2.set_ylabel('SNIP Mean', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # ... 他の統計グラフも同様に実装
        
        plt.tight_layout()
        return fig
    
    def _create_correlation_figure(self, correlations, all_fo, all_snip):
        """相関分析図を生成（PDF用）"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FO vs SNIP Correlation Analysis', fontsize=16)
        
        # ... (前述の相関分析コードと同じ)
        
        plt.tight_layout()
        return fig
    
    def run_full_analysis(self):
        """全分析の実行"""
        print("Starting full AFR score analysis...")
        print("="*50)
        
        # Phase 1: 基本統計
        print("\n[Phase 1] Basic Statistics")
        basic_stats = self.basic_statistics()
        
        # Phase 2: 相関分析
        print("\n[Phase 2] Correlation Analysis")
        correlations = self.correlation_analysis()
        
        print("\n" + "="*50)
        print("Analysis completed!")
        print(f"Results saved to: {self.output_dir}")
        
        return {
            'basic_stats': basic_stats,
            'correlations': correlations
        }

# 使用例
if __name__ == "__main__":
    # PDFで保存する場合
    analyzer = AFRAnalyzer(
        data_dir="weight_scores_llama",
        output_dir="afr_analysis_results",
        save_format='pdf'  # 'png', 'pdf', 'both' から選択
    )
    
    # 全分析実行
    results = analyzer.run_full_analysis()
    
    print("\nSummary:")
    print(f"Global FO-SNIP Pearson correlation: {results['correlations']['aggregated_correlations']['global_pearson']:.4f}")
    print(f"Global FO-SNIP Spearman correlation: {results['correlations']['aggregated_correlations']['global_spearman']:.4f}")
    print(f"Global cosine similarity: {results['correlations']['aggregated_correlations']['global_cosine_similarity']:.4f}")