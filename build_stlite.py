import os
import json
import base64

def get_files_dict(base_dir):
    """
    Reads all relevant files and returns a dictionary mapping 
    relative paths to file contents.
    """
    files = {}
    
    # Files to include
    include_extensions = ['.py', '.csv', '.txt']
    exclude_dirs = ['.git', '__pycache__', '.streamlit']
    
    for root, dirs, filenames in os.walk(base_dir):
        # Mutate dirs in-place to ignore certain directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for filename in filenames:
            if any(filename.endswith(ext) for ext in include_extensions):
                # We skip build_stlite.py itself
                if filename == 'build_stlite.py':
                    continue
                    
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, base_dir).replace('\\', '/')
                
                with open(filepath, 'rb') as f:
                    content = f.read()
                
                # If it's a CSV, encode it as base64 to avoid JSON string escaping issues
                if filename.endswith('.csv'):
                    encoded = base64.b64encode(content).decode('utf-8')
                    files[rel_path] = {
                        "data": f"data:text/csv;base64,{encoded}"
                    }
                else:
                    files[rel_path] = content.decode('utf-8')
                    
    return files

def build_index_html():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = get_files_dict(base_dir)
    
    html_template = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta
      name="viewport"
      content="width=device-width, initial-scale=1, shrink-to-fit=no"
    />
    <title>ISLP Interactive - Stlite</title>
    <!-- Include stlite -->
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.52.0/build/stlite.css"
    />
  </head>
  <body>
    <div id="root"></div>
    <script src="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.52.0/build/stlite.js"></script>
    <script>
      const files = {json.dumps(files)};
      
      // We need to resolve base64 data URIs back to Uint8Array for Stlite to read them correctly as files
      for (const [filepath, content] of Object.entries(files)) {{
          if (content && typeof content === 'object' && content.data) {{
              const base64Data = content.data.split(',')[1];
              const binaryString = window.atob(base64Data);
              const len = binaryString.length;
              const bytes = new Uint8Array(len);
              for (let i = 0; i < len; i++) {{
                  bytes[i] = binaryString.charCodeAt(i);
              }}
              files[filepath] = bytes;
          }}
      }}

      stlite.mount(
        {{
          requirements: ["pandas", "numpy", "matplotlib", "scikit-learn"],
          entrypoint: "app.py",
          files: files,
        }},
        document.getElementById("root")
      );
    </script>
  </body>
</html>
"""
    
    with open(os.path.join(base_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_template)
        
    print("index.html successfully generated for Netlify deployment!")

if __name__ == "__main__":
    build_index_html()
