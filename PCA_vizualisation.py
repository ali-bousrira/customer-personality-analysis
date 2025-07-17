import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

class PCAAnalysisGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PCA Analysis Tool")
        self.root.geometry("1400x900")
        
        # Data storage
        self.df = None
        self.pca_results = None
        
        # Create main interface
        self.create_widgets()
        
    def create_widgets(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Data Loading and Setup
        self.setup_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.setup_tab, text="Data Setup")
        self.create_setup_tab()
        
        # Tab 2: PCA Analysis
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="PCA Analysis")
        self.create_analysis_tab()
        
        # Tab 3: Visualization
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="Visualizations")
        self.create_visualization_tab()
        
        # Tab 4: Component Analysis
        self.component_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.component_tab, text="Component Analysis")
        self.create_component_tab()
        
    def create_setup_tab(self):
        # File loading section
        file_frame = ttk.LabelFrame(self.setup_tab, text="Data Loading", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(file_frame, text="Load CSV File", command=self.load_file).pack(side='left', padx=5)
        ttk.Button(file_frame, text="Load Sample Data", command=self.load_sample_data).pack(side='left', padx=5)
        
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.pack(side='left', padx=20)
        
        # Data preview section
        preview_frame = ttk.LabelFrame(self.setup_tab, text="Data Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for data preview
        self.tree = ttk.Treeview(preview_frame)
        self.tree.pack(fill='both', expand=True)
        
        # Scrollbars for treeview
        tree_scroll_y = ttk.Scrollbar(preview_frame, orient='vertical', command=self.tree.yview)
        tree_scroll_y.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=tree_scroll_y.set)
        
        tree_scroll_x = ttk.Scrollbar(preview_frame, orient='horizontal', command=self.tree.xview)
        tree_scroll_x.pack(side='bottom', fill='x')
        self.tree.configure(xscrollcommand=tree_scroll_x.set)
        
        # Configuration section
        config_frame = ttk.LabelFrame(self.setup_tab, text="PCA Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)
        
        # Columns to exclude
        ttk.Label(config_frame, text="Columns to exclude (comma-separated):").pack(anchor='w')
        self.exclude_entry = ttk.Entry(config_frame, width=50)
        self.exclude_entry.pack(fill='x', pady=2)
        self.exclude_entry.insert(0, "ID")
        
        # Number of components
        comp_frame = ttk.Frame(config_frame)
        comp_frame.pack(fill='x', pady=5)
        
        ttk.Label(comp_frame, text="Number of components:").pack(side='left')
        self.n_components_var = tk.StringVar(value="auto")
        ttk.Radiobutton(comp_frame, text="Auto (90% variance)", variable=self.n_components_var, value="auto").pack(side='left', padx=10)
        ttk.Radiobutton(comp_frame, text="Custom:", variable=self.n_components_var, value="custom").pack(side='left', padx=5)
        self.custom_components = ttk.Entry(comp_frame, width=10)
        self.custom_components.pack(side='left', padx=5)
        
    def create_analysis_tab(self):
        # Analysis controls
        control_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Controls", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="Run PCA Analysis", command=self.run_pca_analysis).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Export Results", command=self.export_results).pack(side='left', padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, wrap='word', height=15)
        self.results_text.pack(fill='both', expand=True)
        
        # Scrollbar for results
        results_scroll = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        results_scroll.pack(side='right', fill='y')
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
    def create_visualization_tab(self):
        # Visualization controls
        viz_control_frame = ttk.LabelFrame(self.viz_tab, text="Visualization Options", padding=10)
        viz_control_frame.pack(fill='x', padx=10, pady=5)
        
        # Plot type selection
        ttk.Label(viz_control_frame, text="Plot Type:").pack(side='left')
        self.plot_type = ttk.Combobox(viz_control_frame, values=[
            "Scree Plot", "Cumulative Variance", "2D Scatter", "3D Scatter", 
            "Biplot", "Correlation Heatmap"
        ], state="readonly", width=15)
        self.plot_type.pack(side='left', padx=5)
        self.plot_type.set("Scree Plot")
        
        # Component selection for scatter plots
        ttk.Label(viz_control_frame, text="PC X:").pack(side='left', padx=(20,5))
        self.pc_x = ttk.Combobox(viz_control_frame, width=8)
        self.pc_x.pack(side='left', padx=5)
        
        ttk.Label(viz_control_frame, text="PC Y:").pack(side='left', padx=5)
        self.pc_y = ttk.Combobox(viz_control_frame, width=8)
        self.pc_y.pack(side='left', padx=5)
        
        ttk.Button(viz_control_frame, text="Generate Plot", command=self.generate_plot).pack(side='left', padx=20)
        
        # Plot display area
        self.plot_frame = ttk.Frame(self.viz_tab)
        self.plot_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
    def create_component_tab(self):
        # Component analysis controls
        comp_control_frame = ttk.LabelFrame(self.component_tab, text="Component Analysis", padding=10)
        comp_control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(comp_control_frame, text="Select Component:").pack(side='left')
        self.component_select = ttk.Combobox(comp_control_frame, width=10)
        self.component_select.pack(side='left', padx=5)
        self.component_select.bind('<<ComboboxSelected>>', self.show_component_loadings)
        
        ttk.Button(comp_control_frame, text="Show Loadings Plot", command=self.plot_component_loadings).pack(side='left', padx=20)
        
        # Component details
        details_frame = ttk.LabelFrame(self.component_tab, text="Component Details", padding=10)
        details_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.component_text = tk.Text(details_frame, wrap='word', height=10)
        self.component_text.pack(fill='both', expand=True)
        
        comp_scroll = ttk.Scrollbar(details_frame, orient='vertical', command=self.component_text.yview)
        comp_scroll.pack(side='right', fill='y')
        self.component_text.configure(yscrollcommand=comp_scroll.set)
        
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.file_label.config(text=f"Loaded: {file_path.split('/')[-1]} ({self.df.shape[0]}x{self.df.shape[1]})")
                self.update_data_preview()
                messagebox.showinfo("Success", f"File loaded successfully!\nShape: {self.df.shape}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def load_sample_data(self):
        """Create sample dataset for demonstration"""
        np.random.seed(42)
        n_samples = 200
        
        # Create correlated features
        base_data = np.random.randn(n_samples, 3)
        
        # Create additional features with known relationships
        data = {
            'Feature1': base_data[:, 0],
            'Feature2': base_data[:, 1],
            'Feature3': base_data[:, 2],
            'Feature4': base_data[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2,
            'Feature5': base_data[:, 1] * 0.7 + base_data[:, 2] * 0.3 + np.random.randn(n_samples) * 0.3,
            'Feature6': base_data[:, 0] - base_data[:, 1] + np.random.randn(n_samples) * 0.4,
            'Feature7': np.random.randn(n_samples),
            'Feature8': base_data[:, 2] * 0.9 + np.random.randn(n_samples) * 0.1,
            'ID': range(1, n_samples + 1)
        }
        
        self.df = pd.DataFrame(data)
        self.file_label.config(text=f"Sample data loaded ({self.df.shape[0]}x{self.df.shape[1]})")
        self.update_data_preview()
        messagebox.showinfo("Success", f"Sample data created!\nShape: {self.df.shape}")
    
    def update_data_preview(self):
        """Update the data preview treeview"""
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if self.df is not None:
            # Configure columns
            self.tree["columns"] = list(self.df.columns)
            self.tree["show"] = "headings"
            
            # Configure column headings
            for col in self.df.columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100)
            
            # Insert data (first 50 rows)
            for index, row in self.df.head(50).iterrows():
                self.tree.insert("", "end", values=list(row))
    
    def run_pca_analysis(self):
        """Run the PCA analysis"""
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            # Get configuration
            exclude_cols = [col.strip() for col in self.exclude_entry.get().split(',') if col.strip()]
            
            n_components = None
            if self.n_components_var.get() == "custom":
                try:
                    n_components = int(self.custom_components.get())
                except ValueError:
                    messagebox.showerror("Error", "Invalid number of components!")
                    return
            
            # Run PCA analysis
            self.pca_results = self.perform_pca_analysis(self.df, n_components, exclude_cols)
            
            # Update UI elements
            self.update_results_display()
            self.update_component_selectors()
            
            messagebox.showinfo("Success", "PCA analysis completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"PCA analysis failed: {str(e)}")
    
    def perform_pca_analysis(self, df, n_components=None, exclude_cols=None):
        """The original PCA analysis function, modified to work with GUI"""
        # Data Preparation
        if exclude_cols is None:
            exclude_cols = ['ID']
        
        df_pca = df.drop(columns=exclude_cols, errors='ignore')
        df_pca = df_pca.fillna(df_pca.median())
        
        numeric_cols = df_pca.select_dtypes(include=[np.number]).columns
        df_pca = df_pca[numeric_cols]
        
        # Standardization
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_pca)
        
        # Determine optimal number of components
        pca_full = PCA()
        pca_full.fit(df_scaled)
        
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        n_80 = np.argmax(cumulative_variance >= 0.80) + 1
        n_90 = np.argmax(cumulative_variance >= 0.90) + 1
        n_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        # Apply PCA
        if n_components is None:
            n_components = n_90
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(df_scaled)
        
        # Create results DataFrame
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        df_pca_result = pd.DataFrame(pca_result, columns=pca_columns, index=df.index)
        
        for col in exclude_cols:
            if col in df.columns:
                df_pca_result[col] = df[col]
        
        return {
            'pca_model': pca,
            'scaler': scaler,
            'pca_data': df_pca_result,
            'original_columns': list(df_pca.columns),
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumulative_variance,
            'components': pca.components_,
            'n_80': n_80,
            'n_90': n_90,
            'n_95': n_95
        }
    
    def update_results_display(self):
        """Update the results text display"""
        self.results_text.delete(1.0, tk.END)
        
        if self.pca_results:
            results = self.pca_results
            
            text = f"""=== PCA Analysis Results ===

Data Preparation:
- Original columns: {len(results['original_columns'])}
- Final dataset shape: {len(results['original_columns'])} features
- Components used: {len(results['explained_variance_ratio'])}

Optimal Components:
- Components for 80% variance: {results['n_80']}
- Components for 90% variance: {results['n_90']}
- Components for 95% variance: {results['n_95']}

Explained Variance:
"""
            
            for i, var in enumerate(results['explained_variance_ratio']):
                text += f"- PC{i+1}: {var:.4f} ({var*100:.2f}%)\n"
            
            text += f"\nCumulative explained variance: {np.sum(results['explained_variance_ratio']):.4f} ({np.sum(results['explained_variance_ratio'])*100:.2f}%)\n"
            
            text += f"\nOriginal Features:\n"
            for i, col in enumerate(results['original_columns']):
                text += f"- {col}\n"
            
            self.results_text.insert(tk.END, text)
    
    def update_component_selectors(self):
        """Update component selector comboboxes"""
        if self.pca_results:
            n_components = len(self.pca_results['explained_variance_ratio'])
            pc_options = [f"PC{i+1}" for i in range(n_components)]
            
            self.pc_x['values'] = pc_options
            self.pc_y['values'] = pc_options
            self.component_select['values'] = pc_options
            
            if len(pc_options) >= 2:
                self.pc_x.set("PC1")
                self.pc_y.set("PC2")
            if len(pc_options) >= 1:
                self.component_select.set("PC1")
    
    def generate_plot(self):
        """Generate the selected plot"""
        if self.pca_results is None:
            messagebox.showerror("Error", "Please run PCA analysis first!")
            return
        
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Create new plot
        fig = Figure(figsize=(12, 8))
        
        plot_type = self.plot_type.get()
        
        if plot_type == "Scree Plot":
            self.create_scree_plot(fig)
        elif plot_type == "Cumulative Variance":
            self.create_cumulative_variance_plot(fig)
        elif plot_type == "2D Scatter":
            self.create_2d_scatter_plot(fig)
        elif plot_type == "Biplot":
            self.create_biplot(fig)
        elif plot_type == "Correlation Heatmap":
            self.create_correlation_heatmap(fig)
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_scree_plot(self, fig):
        ax = fig.add_subplot(111)
        
        explained_var = self.pca_results['explained_variance_ratio']
        components = range(1, len(explained_var) + 1)
        
        ax.plot(components, explained_var, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Scree Plot')
        ax.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, var in enumerate(explained_var):
            ax.annotate(f'{var*100:.1f}%', (i+1, var), textcoords="offset points", xytext=(0,10), ha='center')
        
        fig.tight_layout()
    
    def create_cumulative_variance_plot(self, fig):
        ax = fig.add_subplot(111)
        
        explained_var = self.pca_results['explained_variance_ratio']
        cumulative_var = np.cumsum(explained_var)
        components = range(1, len(explained_var) + 1)
        
        ax.plot(components, cumulative_var, 'ro-', linewidth=2, markersize=8)
        ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='80%')
        ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90%')
        ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95%')
        
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Cumulative Explained Variance Ratio')
        ax.set_title('Cumulative Explained Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
    
    def create_2d_scatter_plot(self, fig):
        ax = fig.add_subplot(111)
        
        pc_x = self.pc_x.get()
        pc_y = self.pc_y.get()
        
        if not pc_x or not pc_y:
            ax.text(0.5, 0.5, 'Please select PC X and PC Y', ha='center', va='center', transform=ax.transAxes)
            return
        
        pca_data = self.pca_results['pca_data']
        
        ax.scatter(pca_data[pc_x], pca_data[pc_y], alpha=0.6)
        ax.set_xlabel(f'{pc_x} ({self.pca_results["explained_variance_ratio"][int(pc_x[2:])-1]*100:.1f}%)')
        ax.set_ylabel(f'{pc_y} ({self.pca_results["explained_variance_ratio"][int(pc_y[2:])-1]*100:.1f}%)')
        ax.set_title(f'PCA: {pc_x} vs {pc_y}')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
    
    def create_biplot(self, fig):
        ax = fig.add_subplot(111)
        
        pca_data = self.pca_results['pca_data']
        components = self.pca_results['components']
        original_cols = self.pca_results['original_columns']
        
        # Scatter plot of data points
        ax.scatter(pca_data['PC1'], pca_data['PC2'], alpha=0.6)
        
        # Add loading vectors
        for i, col in enumerate(original_cols):
            ax.arrow(0, 0, components[0, i]*3, components[1, i]*3, 
                    head_width=0.1, head_length=0.1, fc='red', ec='red')
            ax.text(components[0, i]*3.2, components[1, i]*3.2, col, fontsize=10, ha='center')
        
        ax.set_xlabel(f'PC1 ({self.pca_results["explained_variance_ratio"][0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({self.pca_results["explained_variance_ratio"][1]*100:.1f}%)')
        ax.set_title('PCA Biplot')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
    
    def create_correlation_heatmap(self, fig):
        ax = fig.add_subplot(111)
        
        # Create correlation matrix of original features
        exclude_cols = [col.strip() for col in self.exclude_entry.get().split(',') if col.strip()]
        df_numeric = self.df.drop(columns=exclude_cols, errors='ignore').select_dtypes(include=[np.number])
        
        corr_matrix = df_numeric.corr()
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Correlation')
        
        # Set ticks and labels
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        ax.set_title('Feature Correlation Heatmap')
        
        fig.tight_layout()
    
    def show_component_loadings(self, event=None):
        """Display component loadings in text area"""
        if self.pca_results is None:
            return
        
        selected_pc = self.component_select.get()
        if not selected_pc:
            return
        
        pc_index = int(selected_pc[2:]) - 1
        components = self.pca_results['components']
        original_cols = self.pca_results['original_columns']
        
        self.component_text.delete(1.0, tk.END)
        
        text = f"=== {selected_pc} Loadings ===\n\n"
        text += f"Explained Variance: {self.pca_results['explained_variance_ratio'][pc_index]:.4f} ({self.pca_results['explained_variance_ratio'][pc_index]*100:.2f}%)\n\n"
        text += "Feature Loadings (sorted by absolute value):\n"
        
        # Get loadings and sort by absolute value
        loadings = [(col, components[pc_index, i]) for i, col in enumerate(original_cols)]
        loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for col, loading in loadings:
            text += f"- {col}: {loading:.4f}\n"
        
        self.component_text.insert(tk.END, text)
    
    def plot_component_loadings(self):
        """Create a bar plot of component loadings"""
        if self.pca_results is None:
            messagebox.showerror("Error", "Please run PCA analysis first!")
            return
        
        selected_pc = self.component_select.get()
        if not selected_pc:
            messagebox.showerror("Error", "Please select a component!")
            return
        
        pc_index = int(selected_pc[2:]) - 1
        
        # Create new window for loadings plot
        loadings_window = tk.Toplevel(self.root)
        loadings_window.title(f"{selected_pc} Loadings")
        loadings_window.geometry("800x600")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        components = self.pca_results['components']
        original_cols = self.pca_results['original_columns']
        loadings = components[pc_index, :]
        
        # Sort by absolute value
        sorted_indices = np.argsort(np.abs(loadings))[::-1]
        sorted_cols = [original_cols[i] for i in sorted_indices]
        sorted_loadings = loadings[sorted_indices]
        
        colors = ['red' if x < 0 else 'blue' for x in sorted_loadings]
        bars = ax.bar(range(len(sorted_loadings)), sorted_loadings, color=colors, alpha=0.7)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Loading')
        ax.set_title(f'{selected_pc} Loadings')
        ax.set_xticks(range(len(sorted_cols)))
        ax.set_xticklabels(sorted_cols, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_loadings)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                   f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, loadings_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def export_results(self):
        """Export PCA results to CSV"""
        if self.pca_results is None:
            messagebox.showerror("Error", "No results to export!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save PCA Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.pca_results['pca_data'].to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")

# Main execution
if __name__ == "__main__":
    app = PCAAnalysisGUI()
    app.root.mainloop()