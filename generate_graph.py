import json
import html

def generate_graph_html(papers_file='papers.json', output_file='papers_graph.html'):
    """Generate an interactive force-directed graph from papers JSON."""
    
    with open(papers_file, 'r') as f:
        data = json.load(f)
    
    papers = data['papers']
    categories = data.get('categories', {})
    
    # Build nodes and edges
    nodes = []
    edges = []
    
    for paper in papers:
        # Priority affects visual style
        priority = paper.get('priority', 'medium')
        # Not using size parameter for box shape
        
        # Read status affects visual style
        is_read = paper.get('read', False)
        
        # No label on nodes - title shown on hover and click only
        label = ''
        
        # Get primary category for grouping - ensure it exists
        categories_list = paper.get('categories', [])
        if isinstance(categories_list, list) and len(categories_list) > 0:
            primary_category = categories_list[0]
        else:
            primary_category = 'uncategorized'
        
        nodes.append({
            'id': paper['id'],
            'label': label,
            'title': paper['title'],  # Full title for tooltip and detail view
            'url': paper['url'],
            'categories': categories_list,
            'primary_category': primary_category,
            'notes': paper.get('notes', ''),
            'year': paper.get('year', ''),
            'priority': priority,
            'read': is_read
        })
        
        # Add edges based on related_to (make bidirectional/undirected)
        edge_set = set()  # Track unique edges to avoid duplicates
        for related_id in paper.get('related_to', []):
            # Create a canonical edge representation (sorted tuple)
            edge_key = tuple(sorted([paper['id'], related_id]))
            if edge_key not in edge_set:
                edges.append({'from': paper['id'], 'to': related_id})
                edge_set.add(edge_key)

    # Add invisible edges between papers in same category to force clustering
    papers_by_category = {}
    for paper in papers:
        # For papers with multiple categories, add to all of them
        cats = paper.get('categories', [])
        if not cats:
            cats = ['uncategorized']

        for cat in cats:
            if cat not in papers_by_category:
                papers_by_category[cat] = []
            papers_by_category[cat].append(paper['id'])

    # Create strong connections within each category
    for cat, paper_ids in papers_by_category.items():
        for i, paper_id in enumerate(paper_ids):
            # Connect each paper to a few others in same category
            for j in range(i + 1, min(i + 3, len(paper_ids))):
                edges.append({
                    'from': paper_id,
                    'to': paper_ids[j],
                    'hidden': True,
                    'physics': True,
                    'length': 30  # Very short for tight clustering
                })
    
    # Generate HTML with vis.js
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Research Papers Graph</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            height: 100vh;
            overflow: hidden;
        }}
        .header {{
            margin-bottom: 15px;
        }}
        h1 {{
            margin: 0 0 15px 0;
            font-size: 28px;
        }}
        .controls {{
            margin-bottom: 15px;
            display: none;  /* Hide legends - info shown in detail panel */
        }}
        .main-container {{
            display: flex;
            gap: 20px;
            height: calc(100vh - 140px);
        }}
        .priority-legend {{
            padding: 12px 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #ddd;
            font-size: 16px;
            display: inline-block;
        }}
        .priority-legend strong {{
            margin-right: 15px;
        }}
        .priority-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        .priority-dot {{
            display: inline-block;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
        }}
        .priority-dot.high {{
            width: 20px;
            height: 20px;
            background: #2563eb;
        }}
        .priority-dot.medium {{
            width: 16px;
            height: 16px;
            background: #60a5fa;
        }}
        .priority-dot.low {{
            width: 12px;
            height: 12px;
            background: #bfdbfe;
        }}
        #mynetwork {{
            flex: 1;
            border: 1px solid #ddd;
            background: white;
            border-radius: 8px;
            width: 100%;
        }}
        #info {{
            width: 400px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            border: 1px solid #ddd;
            overflow-y: auto;
            flex-shrink: 0;
        }}
        .paper-link {{
            color: #0066cc;
            text-decoration: none;
        }}
        .paper-link:hover {{
            text-decoration: underline;
        }}
        .priority-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 5px;
        }}
        .priority-high {{ background: #ff6b6b; color: white; }}
        .priority-medium {{ background: #ffd93d; color: #333; }}
        .priority-low {{ background: #e0e0e0; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Reading List</h1>

        <div class="controls">
            <div class="priority-legend">
                <strong>Priority:</strong>
                <span class="priority-item"><span class="priority-dot high"></span>High</span>
                <span class="priority-item"><span class="priority-dot medium"></span>Medium</span>
                <span class="priority-item"><span class="priority-dot low"></span>Low</span>
            </div>
            <div class="priority-legend" style="margin-left: 10px;">
                <strong>Status:</strong>
                <span class="priority-item">━━━ Read</span>
                <span class="priority-item" style="border: 2px; padding: 2px 8px; border-radius: 3px;">┈┈┈ Unread</span>
            </div>
        </div>
    </div>

    <div class="main-container">
        <div id="mynetwork"></div>
        <div id="info">
            <p style="font-size: 16px; color: #666;">Click on a node to see paper details. Double-click to open the paper URL.</p>
        </div>
    </div>

    <script type="text/javascript">
        // Create nodes and edges
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});

        // Create network
        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            autoResize: true,
            nodes: {{
                shape: 'dot',
                size: 8,
                borderWidth: 0,
                borderWidthSelected: 2,
                color: {{
                    border: '#333',
                    background: '#999',
                    highlight: {{
                        border: '#0066cc',
                        background: '#0066cc'
                    }}
                }},
                font: {{
                    size: 0
                }}
            }},
            edges: {{
                width: 1,
                color: {{color: 'rgba(150, 150, 150, 0.3)', highlight: 'rgba(0, 102, 204, 0.6)'}},
                smooth: {{
                    type: 'continuous',
                    forceDirection: 'none',
                    roundness: 0.5
                }},
                arrows: {{
                    to: {{enabled: false}}
                }},
                hidden: false
            }},
            physics: {{
            
                enabled: true,
                forceAtlas2Based: {{
                    gravitationalConstant: -50,
                    centralGravity: 0.005,
                    springLength: 100,
                    springConstant: 0.08,
                    damping: 0.9,
                    avoidOverlap: 0.5
                }},
                maxVelocity: 50,
                minVelocity: 0.1,
                solver: 'forceAtlas2Based',
                stabilization: {{
                    enabled: true,
                    iterations: 3000,
                    updateInterval: 25,
                    fit: true
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 300,
                hideEdgesOnDrag: true,
                hideEdgesOnZoom: false,
                zoomView: true,
                dragView: true,
                hoverConnectedEdges: false
            }}
        }};
        
        var network = new vis.Network(container, data, options);

        setTimeout(() => {{
            network.setOptions({{ physics: {{ enabled: false }} }});
            nodes.get().forEach(n => {{
                nodes.update({{ id: n.id, fixed: {{ x: true, y: true }} }});
            }});
        }}, 2000);
        
        // Style nodes based on category, priority, and status after stabilization
        network.on('stabilized', function() {{
            // Get positions for category labels
            var categoryPositions = {{}};
            var categoryCounts = {{}};
            var categoryBounds = {{}};

            nodes.forEach(function(node) {{
                if (!node.id.startsWith('category_label_')) {{
                    // Track positions for ALL category labels (not just primary)
                    var cats = node.categories || [node.primary_category];
                    var pos = network.getPosition(node.id);

                    cats.forEach(function(cat) {{
                        if (!categoryPositions[cat]) {{
                            categoryPositions[cat] = {{x: 0, y: 0}};
                            categoryCounts[cat] = 0;
                            categoryBounds[cat] = {{minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity}};
                        }}
                        categoryPositions[cat].x += pos.x;
                        categoryPositions[cat].y += pos.y;
                        categoryCounts[cat]++;

                        // Track bounds for label placement
                        categoryBounds[cat].minX = Math.min(categoryBounds[cat].minX, pos.x);
                        categoryBounds[cat].maxX = Math.max(categoryBounds[cat].maxX, pos.x);
                        categoryBounds[cat].minY = Math.min(categoryBounds[cat].minY, pos.y);
                        categoryBounds[cat].maxY = Math.max(categoryBounds[cat].maxY, pos.y);
                    }});
                }}
            }});

            // Calculate bounds of entire graph for position-based coloring
            var allPositions = nodes.get().filter(n => !n.id.startsWith('category_label_')).map(n => network.getPosition(n.id));
            var minX = Math.min(...allPositions.map(p => p.x));
            var maxX = Math.max(...allPositions.map(p => p.x));
            var minY = Math.min(...allPositions.map(p => p.y));
            var maxY = Math.max(...allPositions.map(p => p.y));

            nodes.forEach(function(node) {{
                if (!node.id.startsWith('category_label_')) {{
                    var pos = network.getPosition(node.id);

                    // Color based on position in graph using HSL
                    // Map x position to hue (0-360)
                    var hue = ((pos.x - minX) / (maxX - minX)) * 360;
                    // Map y position to saturation (40-70% for pastel colors)
                    var saturation = 50 + ((pos.y - minY) / (maxY - minY)) * 20;
                    // Fixed lightness for consistent pastel look
                    var lightness = 65;

                    var bgColor = 'hsla(' + hue + ', ' + saturation + '%, ' + lightness + '%, 0.6)';

                    // Subtle size differences based on priority
                    var nodeSize = {{'high': 12, 'medium': 8, 'low': 6}}[node.priority] || 8;

                    // Very subtle border for unread - don't make it too obvious
                    var borderWidth = node.read ? 0 : 1;
                    var borderColor = 'rgba(0, 0, 0, 0.3)';

                    nodes.update({{
                        id: node.id,
                        size: nodeSize,
                        color: {{
                            background: bgColor,
                            border: borderColor,
                            highlight: {{
                                background: bgColor.replace('0.5', '0.8'),
                                border: '#0066cc'
                            }}
                        }},
                        borderWidth: borderWidth,
                        title: node.title  // Show title on hover
                    }});
                }}
            }});

            // Add category labels in the CENTER of each cluster with collision detection
            var labelPositions = [];
            var minLabelDistance = 120; // Minimum distance between label centers (increased)

            Object.keys(categoryPositions).forEach(function(cat) {{
                var avgX = categoryPositions[cat].x / categoryCounts[cat];
                var avgY = categoryPositions[cat].y / categoryCounts[cat];

                // Scale font size based on number of papers in category
                // Base size 18, up to 28 for large categories
                var categorySize = categoryCounts[cat];
                var fontSize = Math.min(28, 18 + Math.sqrt(categorySize) * 2);

                // Check for collisions with existing labels and adjust position
                var adjusted = false;
                var attempts = 0;
                var finalX = avgX;
                var finalY = avgY;

                while (!adjusted && attempts < 50) {{
                    var hasCollision = false;

                    for (var i = 0; i < labelPositions.length; i++) {{
                        var dx = finalX - labelPositions[i].x;
                        var dy = finalY - labelPositions[i].y;
                        var distance = Math.sqrt(dx * dx + dy * dy);

                        // Account for label sizes when checking collision
                        var minDist = minLabelDistance + (fontSize + labelPositions[i].fontSize) / 4;

                        if (distance < minDist) {{
                            hasCollision = true;
                            // Push label away from collision at a perpendicular angle
                            var angle = Math.atan2(dy, dx);
                            // Add some randomness to avoid getting stuck
                            var offsetAngle = angle + (attempts > 10 ? (Math.random() - 0.5) * 0.5 : 0);
                            finalX = labelPositions[i].x + Math.cos(offsetAngle) * minDist * 1.1;
                            finalY = labelPositions[i].y + Math.sin(offsetAngle) * minDist * 1.1;
                            break;
                        }}
                    }}

                    if (!hasCollision) {{
                        adjusted = true;
                    }}
                    attempts++;
                }}

                labelPositions.push({{x: finalX, y: finalY, category: cat, fontSize: fontSize}});

                var labelId = 'category_label_' + cat;
                var labelText = cat.toUpperCase();

                try {{
                    nodes.add({{
                        id: labelId,
                        label: labelText,
                        x: finalX,
                        y: finalY,
                        fixed: {{x: true, y: true}},
                        shape: 'text',
                        font: {{
                            size: fontSize,
                            color: 'rgba(0, 0, 0, 1)',
                            face: '-apple-system, BlinkMacSystemFont, sans-serif',
                            bold: true,
                            strokeWidth: 0
                        }},
                        physics: false,
                        chosen: false,
                        interaction: false,
                        mass: 1,
                        value: undefined
                    }});
                }} catch(e) {{
                    // Label already exists, update position
                    try {{
                        nodes.update({{
                            id: labelId,
                            x: finalX,
                            y: finalY,
                            interaction: false
                        }});
                    }} catch(e2) {{}}
                }}
            }});
        }});
        
        // Handle node clicks - filter out category labels

        network.on('click', function (params) {{
    if (params.pointer && params.pointer.canvas) {{
        const clickPos = params.pointer.canvas;
        const clickedNodes = [];

        nodes.get().forEach(n => {{
            const pos = network.getPosition(n.id);
            if (!pos) return;
            const dx = pos.x - clickPos.x;
            const dy = pos.y - clickPos.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            // Adjust 20 if your nodes are larger/smaller
            if (dist < 20 && !n.id.startsWith('category_label_')) clickedNodes.push(n.id);
        }});

        if (clickedNodes.length > 0) {{
            // Include both real and label nodes
            clickedNodes.forEach(id => {{
                const node = nodes.get(id);
                displayPaperInfo(node);
            }});
        }}
    }}
}});

        
        // Handle double-click to open URL
  

    network.on('doubleClick', function (params) {{
    if (params.pointer && params.pointer.canvas) {{
        const clickPos = params.pointer.canvas;
        const clickedNodes = [];

        nodes.get().forEach(n => {{
            const pos = network.getPosition(n.id);
            if (!pos) return;
            const dx = pos.x - clickPos.x;
            const dy = pos.y - clickPos.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            // Adjust 20 if your nodes are larger/smaller
            if (dist < 20 && !n.id.startsWith('category_label_') && n.url) {{
                window.open(n.url, '_blank');
                return;
            }}
        }});
    }}
}});
        
        function displayPaperInfo(node) {{
            var infoDiv = document.getElementById('info');

            // Display ALL categories - categories is stored as an array
            var categoriesHtml = '';
            console.log('Node data:', node);
            console.log('Categories:', node.categories);

            if (node.categories && node.categories.length > 0) {{
                var cats = node.categories;
                if (typeof cats === 'string') {{
                    cats = [cats];
                }}
                categoriesHtml = cats.map(cat =>
                    '<span style="background: #e0e0e0; padding: 6px 12px; border-radius: 4px; margin-right: 8px; margin-bottom: 6px; display: inline-block; font-size: 15px;">' +
                    (cat.charAt(0).toUpperCase() + cat.slice(1).replace(/-/g, ' ')) +
                    '</span>'
                ).join('');
            }} else {{
                categoriesHtml = '<span style="color: #999;">No categories (debug: ' + JSON.stringify(node.categories) + ')</span>';
            }}

            var priorityClass = 'priority-' + (node.priority || 'medium');
            var priorityBadge = '<span class="priority-badge ' + priorityClass + '">Priority: ' +
                (node.priority || 'medium').toUpperCase() + '</span>';

            var readBadge = node.read
                ? '<span style="background: #10b981; color: white; padding: 4px 10px; border-radius: 4px; font-size: 14px; font-weight: 600; margin-left: 8px;">✓ READ</span>'
                : '<span style="background: #f59e0b; color: white; padding: 4px 10px; border-radius: 4px; font-size: 14px; font-weight: 600; margin-left: 8px;">UNREAD</span>';

            infoDiv.innerHTML = `
                <h2 style="margin-top: 0; font-size: 20px; line-height: 1.4;">${{escapeHtml(node.title)}} ${{priorityBadge}} ${{readBadge}}</h2>
                <p style="font-size: 16px; margin-bottom: 12px;"><strong>Categories:</strong></p>
                <div style="margin-bottom: 15px;">${{categoriesHtml}}</div>
                <p style="font-size: 16px;"><strong>Year:</strong> ${{node.year || 'N/A'}}</p>
                ${{node.notes ? '<p style="font-size: 16px;"><strong>Notes:</strong> ' + escapeHtml(node.notes) + '</p>' : ''}}
                <p style="font-size: 16px;"><a href="${{node.url}}" target="_blank" class="paper-link">Open Paper →</a></p>
            `;
        }}
        
        function escapeHtml(text) {{
            var map = {{
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            }};
            return text.replace(/[&<>"']/g, function(m) {{ return map[m]; }});
        }}
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_template)
    
    print(f"✓ Generated {output_file}")
    print(f"  Total papers: {len(papers)}")
    print(f"  Total connections: {len(edges)}")

def generate_legend(categories):
    """Generate HTML for category legend."""
    items = []
    for cat_id, cat_data in categories.items():
        color = cat_data.get('color', '#999')
        label = cat_data.get('label', cat_id)
        items.append(f'<div class="legend-item"><span class="legend-color" style="background: {color};"></span>{label}</div>')
    return '\n'.join(items)

if __name__ == '__main__':
    generate_graph_html()