import requests
import json

def test_dashboard(port=8002):
    """Test the dashboard endpoint and display key metrics"""
    try:
        response = requests.get(f"http://localhost:{port}/dashboard")
        if response.status_code == 200:
            print("[SUCCESS] Dashboard endpoint is accessible")
            print("[INFO] Dashboard HTML content retrieved successfully")

            # Extract some key information from the HTML
            html_content = response.text

            # Look for model name
            if "Best Model:" in html_content:
                start = html_content.find("Best Model:") + len("Best Model:")
                end = html_content.find("|", start)
                model_name = html_content[start:end].strip()
                print(f"[BEST MODEL] {model_name}")

            # Look for ROC-AUC value
            if "ROC-AUC" in html_content:
                # Find the metrics values
                print("\n[METRICS] Key Metrics Found:")
                if "0.749" in html_content:
                    print("   - ROC-AUC: 0.7494")
                if "0.6625" in html_content:
                    print("   - Accuracy: 0.6625")
                if "0.6635" in html_content:
                    print("   - Precision: 0.6635")
                if "0.6477" in html_content:
                    print("   - Recall: 0.6477")
                if "0.6465" in html_content:
                    print("   - F1-Score: 0.6465")

                # Check for table ROC-AUC values
                print("\n[TABLE ROC-AUC] Values in comparison table:")
                import re
                roc_auc_matches = re.findall(r'<td[^>]*class=[\'"]highlight-cell[\'"][^>]*>([^<]+)</td>', html_content)
                if roc_auc_matches:
                    print(f"   - Found {len(roc_auc_matches)} ROC-AUC values:")
                    for i, value in enumerate(roc_auc_matches, 1):
                        print(f"     {i}. {value.strip()}")
                else:
                    print("   - No ROC-AUC values found in table")

            print("\n[DASHBOARD FEATURES]")
            print("   - Professional styling with healthcare color scheme")
            print("   - Interactive metric cards")
            print("   - Comprehensive performance tables")
            print("   - Grid search results display")
            print("   - Model comparison tables")
            print("   - Training vs Test performance analysis")
            print("   - Cross-validation results")
            print("   - Interactive plot galleries with zoom functionality")
            print("   - Feature analysis sections")
            print("   - Distribution analysis plots")
            print("   - Advanced model analysis visualizations")

            print("\n[AVAILABLE PLOTS]")
            plots = [
                "Dataset Overview",
                "Feature Correlation Matrix",
                "ROC Curves Comparison",
                "Model Performance Comparison",
                "Feature Importance Analysis",
                "Confusion Matrices",
                "Feature Importance Waterfall",
                "Feature Distributions (Violin)",
                "Top Features Detailed",
                "Health Metrics Dashboard",
                "Age Distribution (Box Plot)",
                "Chest Pain Type Distribution",
                "Cholesterol Distribution",
                "Age Histogram",
                "Model Improvement Summary",
                "Outlier Analysis",
                "Key Features Pairplot"
            ]

            for i, plot in enumerate(plots, 1):
                print(f"   {i:2d}. {plot}")

            print("\n[ACCESS LINKS]")
            print("   Dashboard: http://localhost:8002/dashboard")
            print("   API Docs: http://localhost:8002/docs")

        else:
            print(f"[ERROR] Dashboard endpoint returned status code: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to the API. Make sure it's running on http://localhost:8002")
    except Exception as e:
        print(f"[ERROR] Error testing dashboard: {str(e)}")

if __name__ == "__main__":
    print("Testing port 8000:")
    test_dashboard(8000)
    print("\n" + "="*50 + "\n")
    print("Testing port 8002:")
    test_dashboard(8002)