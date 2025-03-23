import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

def load_network_data(file_path="job_network.json"):
    """Load network data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        print("Please run view_job_relationships.py first and export the network data.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        sys.exit(1)

def create_graph(network_data):
    """Create a NetworkX graph from the network data"""
    G = nx.Graph()
    
    # Add nodes with attributes
    for node in network_data.get("nodes", []):
        G.add_node(
            node["id"],
            name=node.get("name", ""),
            node_type=node.get("type", ""),
            category=node.get("category", ""),
            entity_type=node.get("entity_type", ""),
            company=node.get("company", "")
        )
    
    # Add edges with attributes
    for link in network_data.get("links", []):
        G.add_edge(
            link["source"],
            link["target"],
            type=link.get("type", "")
        )
    
    return G

def get_node_colors(G):
    """Assign colors to nodes based on type and category"""
    # Create color maps
    node_type_colors = {
        "job": "lightblue",
        "skill": "lightgreen",
        "entity": "salmon"
    }
    
    # Skill category colors
    category_colors = {
        "PROGRAMMING": "forestgreen",
        "WEB_TECH": "limegreen",
        "DATA_SCIENCE": "mediumseagreen",
        "DATABASE": "darkseagreen",
        "CLOUD": "turquoise",
        "AI_LLM": "mediumaquamarine",
        "SOFT_SKILLS": "palegreen"
    }
    
    # Entity type colors
    entity_type_colors = {
        "WORKS_AT": "coral",
        "LOCATED_IN": "lightcoral"
    }
    
    # Create color map
    colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get("node_type", "")
        
        # Default color
        color = "gray"
        
        # Set color based on node type
        if node_type == "job":
            color = node_type_colors.get("job", "gray")
        elif node_type == "skill":
            category = G.nodes[node].get("category", "")
            color = category_colors.get(category, node_type_colors.get("skill", "gray"))
        elif node_type == "entity":
            entity_type = G.nodes[node].get("entity_type", "")
            color = entity_type_colors.get(entity_type, node_type_colors.get("entity", "gray"))
        
        colors.append(color)
    
    return colors

def get_node_sizes(G):
    """Determine node sizes based on connectivity"""
    sizes = []
    for node in G.nodes():
        node_type = G.nodes[node].get("node_type", "")
        degree = G.degree(node)
        
        # Default size
        size = 100
        
        # Set size based on node type and degree
        if node_type == "job":
            size = 300  # Jobs are larger
        elif node_type == "skill":
            size = 50 + (degree * 15)  # Skills scale with connections
        elif node_type == "entity":
            size = 50 + (degree * 20)  # Entities scale with connections
        
        sizes.append(size)
    
    return sizes

def create_network_visualization(G, output_file="job_network_graph.png"):
    """Create a visualization of the job network"""
    # Set up the figure
    plt.figure(figsize=(16, 12))
    
    # Node colors
    node_colors = get_node_colors(G)
    
    # Node sizes
    node_sizes = get_node_sizes(G)
    
    # Set up the layout
    layout = nx.spring_layout(G, k=0.4, iterations=50)
    
    # Draw the network
    nx.draw_networkx(
        G,
        pos=layout,
        with_labels=False,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="gray",
        alpha=0.8,
        width=0.5
    )
    
    # Add labels for important nodes (jobs and high-degree nodes)
    labels = {}
    for node in G.nodes():
        if G.nodes[node].get("node_type") == "job":
            labels[node] = G.nodes[node].get("name")
        elif G.degree(node) > 2:  # Only label nodes with multiple connections
            labels[node] = G.nodes[node].get("name")
    
    # Draw labels for important nodes
    nx.draw_networkx_labels(
        G,
        pos=layout,
        labels=labels,
        font_size=6,
        font_family="sans-serif"
    )
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="lightblue", markersize=10, label='Job'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="forestgreen", markersize=10, label='Programming Skill'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="mediumseagreen", markersize=10, label='Data Science Skill'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="coral", markersize=10, label='Company'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="lightcoral", markersize=10, label='Location')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set title
    plt.title("Job Network Analysis: Skills and Relationships")
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Network visualization saved to {output_file}")
    
    # Show the visualization
    plt.show()

def analyze_network(G):
    """Analyze the network and print statistics"""
    print("\n=== Network Analysis ===")
    
    # Basic network statistics
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Node type counts
    node_types = defaultdict(int)
    for node in G.nodes():
        node_type = G.nodes[node].get("node_type", "unknown")
        node_types[node_type] += 1
    
    print("\nNode types:")
    for node_type, count in node_types.items():
        print(f"  - {node_type}: {count}")
    
    # Skill categories
    if node_types["skill"] > 0:
        skill_categories = defaultdict(int)
        for node in G.nodes():
            if G.nodes[node].get("node_type") == "skill":
                category = G.nodes[node].get("category", "unknown")
                skill_categories[category] += 1
        
        print("\nSkill categories:")
        for category, count in skill_categories.items():
            print(f"  - {category}: {count}")
    
    # Most connected nodes
    degree_centrality = nx.degree_centrality(G)
    top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\nMost connected nodes:")
    for node, centrality in top_nodes:
        node_name = G.nodes[node].get("name", node)
        node_type = G.nodes[node].get("node_type", "unknown")
        print(f"  - {node_name} ({node_type}): {centrality:.4f}")
    
    # Network density
    density = nx.density(G)
    print(f"\nNetwork density: {density:.4f}")
    
    # Average clustering coefficient
    avg_clustering = nx.average_clustering(G)
    print(f"Average clustering coefficient: {avg_clustering:.4f}")

def main():
    """Main function"""
    # Get input file path
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "job_network.json"
    
    # Load network data
    print(f"Loading network data from {input_file}...")
    network_data = load_network_data(input_file)
    
    # Create network graph
    G = create_graph(network_data)
    
    # Analyze network
    analyze_network(G)
    
    # Visualize network
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = "job_network_graph.png"
    
    # Create visualization
    create_network_visualization(G, output_file)

if __name__ == "__main__":
    try:
        # Check if required packages are installed
        import networkx
        import matplotlib
        main()
    except ImportError:
        print("This script requires the 'networkx' and 'matplotlib' packages.")
        print("Please install them with: pip install networkx matplotlib")
        sys.exit(1) 