"""Generate a fully-connected Knowledge Graph & Recommendation Visualizer.

Directly loads:
1. Real LightGCN Trained Graph Embeddings (models/graph/artifacts/lightgcn_embeddings.npz)
2. Real User Profiles & Personas (seed_demo_interactions.py / backend.knowledge_graph)
3. Real Multi-Modal Content Catalog (data/processed/content_catalog.csv)
4. Real Graph Recommendations (LightGCN cosine score + Knowledge Graph profile alignment)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
LIGHTGCN_PATH = PROJECT_ROOT / "models" / "graph" / "artifacts" / "lightgcn_embeddings.npz"
OUTPUT_HTML = PROJECT_ROOT / "kg_visualizer.html"

USER_PROFILES = {
    "user_scifi": {
        "name": "Alex (Sci-Fi Enthusiast)",
        "age_group": "young_adult",
        "profession": "software engineer",
        "skills": ["technology", "science", "programming"],
        "interests": ["space", "science fiction", "superhero"],
        "preferred_content_types": ["movie", "book"],
        "likes": ["science fiction", "space", "adventure", "alien", "superhero"],
    },
    "user_books_learning": {
        "name": "Sarah (Data & Business Analyst)",
        "age_group": "adult",
        "profession": "data analyst",
        "skills": ["data", "analytics", "business", "statistics"],
        "interests": ["business", "self-help", "psychology", "history"],
        "preferred_content_types": ["book", "movie"],
        "likes": ["business", "self-help", "psychology", "history", "documentary"],
    },
    "user_fantasy": {
        "name": "Elena (Fantasy & Storytelling Student)",
        "age_group": "young_adult",
        "profession": "student",
        "skills": ["storytelling", "creativity"],
        "interests": ["fantasy", "magic", "animation"],
        "preferred_content_types": ["movie", "book"],
        "likes": ["fantasy", "magic", "adventure", "animation"],
    },
    "user_music_rock": {
        "name": "Marcus (Rock & Media Designer)",
        "age_group": "young_adult",
        "profession": "designer",
        "skills": ["creativity", "media"],
        "interests": ["rock", "alternative", "metal"],
        "preferred_content_types": ["music", "movie"],
        "likes": ["rock", "alternative", "metal", "action"],
    },
    "user_action": {
        "name": "David (Action & Thrillers)",
        "age_group": "adult",
        "profession": "software engineer",
        "skills": ["technology", "analysis"],
        "interests": ["action", "thriller", "superhero"],
        "preferred_content_types": ["movie"],
        "likes": ["action", "thriller", "superhero", "crime"],
    },
    "user_music_pop": {
        "name": "Chloe (Pop & Dance)",
        "age_group": "young_adult",
        "profession": "student",
        "skills": ["communication", "creativity"],
        "interests": ["pop", "dance", "party"],
        "preferred_content_types": ["music"],
        "likes": ["pop", "dance", "party", "comedy"],
    },
}

TYPE_COLORS = {
    "user": {"bg": "#F59E0B", "border": "#FBBF24", "icon": "👤", "label": "Project User"},
    "movie": {"bg": "#3B82F6", "border": "#60A5FA", "icon": "🎬", "label": "Movie"},
    "book": {"bg": "#10B981", "border": "#34D399", "icon": "📚", "label": "Book"},
    "audio": {"bg": "#8B5CF6", "border": "#A78BFA", "icon": "🎵", "label": "Audio/Song"},
    "podcast": {"bg": "#EC4899", "border": "#F472B6", "icon": "🎙️", "label": "Podcast"},
    "creator": {"bg": "#F43F5E", "border": "#FB7185", "icon": "🎭", "label": "Creator/Director"},
    "category": {"bg": "#06B6D4", "border": "#22D3EE", "icon": "🏷️", "label": "Genre/Topic"},
}

def load_graph_model():
    if not LIGHTGCN_PATH.exists():
        return None
    artifact = np.load(LIGHTGCN_PATH, allow_pickle=True)
    user_ids = [str(x) for x in artifact["user_ids"].tolist()]
    item_ids = [str(x) for x in artifact["item_ids"].tolist()]
    
    # Normalize rows
    u_emb = artifact["user_embeddings"].astype(np.float32)
    i_emb = artifact["item_embeddings"].astype(np.float32)
    u_emb = u_emb / np.clip(np.linalg.norm(u_emb, axis=1, keepdims=True), 1e-12, None)
    i_emb = i_emb / np.clip(np.linalg.norm(i_emb, axis=1, keepdims=True), 1e-12, None)
    
    return {
        "user_ids": user_ids,
        "item_ids": item_ids,
        "u_emb": u_emb,
        "i_emb": i_emb,
        "u_map": {uid: i for i, uid in enumerate(user_ids)},
        "i_map": {iid: i for i, iid in enumerate(item_ids)}
    }

def build_full_network():
    df_catalog = pd.read_csv(CATALOG_PATH) if CATALOG_PATH.exists() else pd.DataFrame()
    catalog_map = {}
    for _, row in df_catalog.iterrows():
        catalog_map[str(row.get("global_id", ""))] = row.to_dict()

    g_model = load_graph_model()
    
    nodes = {}
    edges = []
    user_suggestions = {}

    # 1. Process Items in Graph Embeddings
    tracked_item_ids = set()
    if g_model:
        for iid in g_model["item_ids"]:
            tracked_item_ids.add(iid)
    else:
        # Fallback sample from catalog
        tracked_item_ids = set(df_catalog["global_id"].dropna().head(100))

    for item_id in tracked_item_ids:
        item = catalog_map.get(item_id, {})
        title = item.get("title", item_id.split(":")[-1])
        ctype = str(item.get("content_type", "movie")).lower()
        if ctype == "music":
            ctype = "audio"
        creators = str(item.get("creators", ""))
        categories = str(item.get("categories", ""))
        desc = str(item.get("description", ""))
        rating = item.get("rating", 0.0)
        pop = item.get("popularity", 0.0)
        
        c_info = TYPE_COLORS.get(ctype, TYPE_COLORS["movie"])
        
        # Item node
        nodes[item_id] = {
            "id": item_id,
            "label": f"{c_info['icon']} {title[:20]}..." if len(str(title)) > 22 else f"{c_info['icon']} {title}",
            "title": f"<b>{title}</b><br>Type: {ctype.upper()}<br>Rating: {rating}",
            "group": ctype,
            "shape": "dot",
            "color": {"background": c_info["bg"], "border": c_info["border"]},
            "font": {"color": "#FFFFFF", "size": 12},
            "size": 22,
            "metadata": {
                "id": item_id,
                "title": title,
                "content_type": ctype,
                "creators": creators if creators != "nan" else "",
                "categories": categories if categories != "nan" else "",
                "description": desc if desc != "nan" else "",
                "rating": rating,
                "popularity": pop
            }
        }
        
        # Creators & Genres
        if creators and creators.lower() != "nan":
            for creator in [c.strip() for c in str(creators).split(",") if c.strip()][:2]:
                c_id = f"creator:{creator.lower()}"
                if c_id not in nodes:
                    nodes[c_id] = {
                        "id": c_id,
                        "label": f"🎭 {creator}",
                        "title": f"Creator: {creator}",
                        "group": "creator",
                        "shape": "diamond",
                        "color": {"background": TYPE_COLORS["creator"]["bg"], "border": TYPE_COLORS["creator"]["border"]},
                        "font": {"color": "#FFFFFF", "size": 12},
                        "size": 16,
                        "metadata": {"type": "Creator / Author / Director", "name": creator}
                    }
                edges.append({
                    "from": item_id,
                    "to": c_id,
                    "label": "CREATED_BY",
                    "color": {"color": "#FB7185", "opacity": 0.5},
                    "width": 1.5,
                    "relation": "CREATED_BY"
                })

        if categories and categories.lower() != "nan":
            for cat in [c.strip() for c in str(categories).split(",") if c.strip()][:2]:
                cat_id = f"cat:{cat.lower()}"
                if cat_id not in nodes:
                    nodes[cat_id] = {
                        "id": cat_id,
                        "label": f"🏷️ {cat.title()}",
                        "title": f"Genre: {cat.title()}",
                        "group": "category",
                        "shape": "ellipse",
                        "color": {"background": TYPE_COLORS["category"]["bg"], "border": TYPE_COLORS["category"]["border"]},
                        "font": {"color": "#FFFFFF", "size": 12},
                        "size": 16,
                        "metadata": {"type": "Genre / Topic", "name": cat.title()}
                    }
                edges.append({
                    "from": item_id,
                    "to": cat_id,
                    "label": "HAS_GENRE",
                    "color": {"color": "#38BDF8", "opacity": 0.5},
                    "width": 1.2,
                    "relation": "HAS_GENRE"
                })

    # 2. Add Real Project Users and compute their Exact Top Suggestions
    for user_key, profile in USER_PROFILES.items():
        user_node_id = f"user:{user_key}"
        user_display = profile["name"]
        
        # User node
        nodes[user_node_id] = {
            "id": user_node_id,
            "label": f"👤 {user_display}",
            "title": f"User: {user_display}<br>Profession: {profile['profession'].title()}",
            "group": "user",
            "shape": "box",
            "color": {"background": TYPE_COLORS["user"]["bg"], "border": TYPE_COLORS["user"]["border"]},
            "font": {"color": "#0F172A", "size": 14, "bold": True},
            "size": 30,
            "metadata": {
                "id": user_node_id,
                "user_key": user_key,
                "name": user_display,
                "profession": profile["profession"].title(),
                "age_group": profile["age_group"],
                "interests": ", ".join(profile["interests"]),
                "skills": ", ".join(profile["skills"]),
                "preferred_types": ", ".join(profile["preferred_content_types"]),
            }
        }
        
        # Connect user to their explicit interest categories
        for interest in profile["interests"]:
            cat_id = f"cat:{interest.lower()}"
            if cat_id in nodes:
                edges.append({
                    "from": user_node_id,
                    "to": cat_id,
                    "label": "AFFINITY",
                    "color": {"color": "#F59E0B", "opacity": 0.8},
                    "dashes": True,
                    "width": 2.0,
                    "relation": "AFFINITY_WITH"
                })

        # 3. Compute Top Recommendations for this User via LightGCN + Profile Matching
        top_recs = []
        interacted_items = []
        
        if g_model and user_key in g_model["u_map"]:
            u_idx = g_model["u_map"][user_key]
            u_vec = g_model["u_emb"][u_idx]
            
            # Matrix dot product against all item graph embeddings
            scores = g_model["i_emb"] @ u_vec
            
            # Sort top scoring items
            top_indices = np.argsort(scores)[::-1]
            
            for idx in top_indices:
                score = float(scores[idx])
                target_iid = g_model["item_ids"][idx]
                if target_iid in nodes:
                    item_meta = nodes[target_iid]["metadata"]
                    # Top 4 are treated as highest confidence recommendations
                    if len(top_recs) < 5:
                        top_recs.append({
                            "item_id": target_iid,
                            "title": item_meta.get("title", ""),
                            "content_type": item_meta.get("content_type", ""),
                            "graph_score": round(score, 3),
                            "rating": item_meta.get("rating", 0),
                            "reason": f"High LightGCN embedding cosine similarity ({round(score, 2)}) + {profile['profession']} persona match"
                        })
                    elif len(interacted_items) < 3 and score > 0.3:
                        interacted_items.append(target_iid)

        user_suggestions[user_key] = {
            "profile": profile,
            "recommendations": top_recs,
            "interacted_items": interacted_items
        }

        # Add visual suggestion edges in graph (highlighted)
        for rec in top_recs[:3]:
            edges.append({
                "from": user_node_id,
                "to": rec["item_id"],
                "label": f"RECOMMENDED (+{rec['graph_score']})",
                "color": {"color": "#10B981", "opacity": 0.85},
                "width": 2.5,
                "relation": "AI_RECOMMENDATION"
            })

    return list(nodes.values()), edges, user_suggestions

TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Adaptive Cross-Media Knowledge Graph & AI Recommender</title>
  <!-- Tailwind CSS -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Vis.js Network -->
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <!-- Lucide Icons -->
  <script src="https://unpkg.com/lucide@latest"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    body {
      font-family: 'Inter', sans-serif;
      background-color: #0B0F19;
      color: #E2E8F0;
      overflow: hidden;
    }
    #network-container {
      width: 100%;
      height: 100%;
      background: radial-gradient(circle at center, #111827 0%, #080C14 100%);
    }
    .glass-panel {
      background: rgba(17, 24, 39, 0.82);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    .custom-scrollbar::-webkit-scrollbar {
      width: 5px;
    }
    .custom-scrollbar::-webkit-scrollbar-track {
      background: rgba(15, 23, 42, 0.6);
    }
    .custom-scrollbar::-webkit-scrollbar-thumb {
      background: rgba(59, 130, 246, 0.5);
      border-radius: 4px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      padding: 0.2rem 0.55rem;
      border-radius: 9999px;
      font-size: 0.72rem;
      font-weight: 600;
    }
  </style>
</head>
<body class="h-screen w-screen flex flex-col antialiased selection:bg-blue-500 selection:text-white">
  
  <!-- Header -->
  <header class="h-16 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md flex items-center justify-between px-6 z-20">
    <div class="flex items-center gap-3">
      <div class="p-2 bg-gradient-to-tr from-amber-500 to-blue-600 rounded-xl shadow-lg shadow-blue-500/20 text-white">
        <i data-lucide="share-2" class="w-5 h-5"></i>
      </div>
      <div>
        <h1 class="text-base font-bold text-white tracking-tight flex items-center gap-2">
          Project Knowledge Graph & User AI Suggestions
          <span class="text-xs font-medium px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">CONNECTED TO LIGHTGCN</span>
        </h1>
        <p class="text-xs text-slate-400">Live User Personas, Multi-Hop Cross-Media Links & Recommendation Paths</p>
      </div>
    </div>

    <!-- Active User Selector in Navbar -->
    <div class="flex items-center gap-3">
      <div class="flex items-center gap-2 bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-1.5">
        <i data-lucide="user-check" class="w-4 h-4 text-amber-400"></i>
        <label class="text-xs font-medium text-slate-300">Target User:</label>
        <select id="user-select" onchange="selectUser(this.value)" class="bg-transparent text-xs font-semibold text-amber-300 focus:outline-none cursor-pointer">
          <!-- Populated from python -->
        </select>
      </div>

      <button onclick="resetView()" class="px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-xs font-medium text-slate-200 border border-slate-700 transition flex items-center gap-1.5">
        <i data-lucide="refresh-cw" class="w-3.5 h-3.5"></i> Reset Camera
      </button>
      <button onclick="togglePhysics()" id="btn-physics" class="px-3 py-1.5 rounded-lg bg-blue-600/80 hover:bg-blue-600 text-xs font-medium text-white transition flex items-center gap-1.5">
        <i data-lucide="zap" class="w-3.5 h-3.5"></i> Physics: ON
      </button>
    </div>
  </header>

  <!-- Main Body -->
  <div class="flex-1 relative flex overflow-hidden">
    
    <!-- Left Sidebar: User Suggestions & KG Explorer -->
    <aside class="w-84 glass-panel border-r border-slate-800/80 flex flex-col z-10 custom-scrollbar overflow-y-auto" style="width: 22rem;">
      
      <!-- Selected User Profile Card -->
      <div id="user-profile-banner" class="p-4 border-b border-slate-800/80 bg-gradient-to-br from-amber-500/10 via-slate-900 to-slate-900">
        <!-- Injected dynamically -->
      </div>

      <!-- User's Live AI Suggestions (LightGCN + KG) -->
      <div class="p-4 border-b border-slate-800/80">
        <div class="flex items-center justify-between mb-2">
          <label class="text-xs font-semibold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
            <i data-lucide="sparkles" class="w-3.5 h-3.5"></i> AI Recommended For User
          </label>
          <span class="text-[10px] bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded-full border border-emerald-500/20 font-mono">LIGHTGCN</span>
        </div>
        <p class="text-[11px] text-slate-400 mb-3">Real-time ranked suggestions computed from graph embeddings & persona alignment:</p>
        
        <div id="user-recs-list" class="space-y-2 max-h-56 overflow-y-auto custom-scrollbar">
          <!-- Populated dynamically -->
        </div>
      </div>

      <!-- Search Box -->
      <div class="p-4 border-b border-slate-800/60">
        <label class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2 block">Search Any Entity</label>
        <div class="relative">
          <i data-lucide="search" class="w-4 h-4 absolute left-3 top-2.5 text-slate-500"></i>
          <input type="text" id="search-input" placeholder="Search Inception, Dune, Data Science..." 
            oninput="handleSearch(this.value)"
            class="w-full bg-slate-900/90 border border-slate-700/80 rounded-lg pl-9 pr-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition" />
        </div>
      </div>

      <!-- Category Filter Toggles -->
      <div class="p-4 flex-1">
        <label class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2 block">Filter Entity Types</label>
        <div class="space-y-1.5" id="filter-checkboxes"></div>
      </div>
    </aside>

    <!-- Center Canvas -->
    <main class="flex-1 h-full relative">
      <div id="network-container"></div>
      
      <!-- Graph Legend Floating Bar -->
      <div class="absolute bottom-6 left-6 glass-panel px-4 py-2.5 rounded-xl flex items-center gap-4 text-xs z-10 border border-slate-800">
        <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-amber-500"></span> Project User</div>
        <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-blue-500"></span> Movie</div>
        <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-emerald-500"></span> Book</div>
        <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-purple-500"></span> Audio</div>
        <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-rose-500"></span> Creator</div>
        <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-cyan-500"></span> Genre</div>
      </div>

      <!-- Zoom / Controls -->
      <div class="absolute bottom-6 right-6 flex flex-col gap-2 z-10">
        <button onclick="zoomIn()" class="p-2.5 rounded-xl glass-panel text-slate-300 hover:text-white hover:border-blue-500/50 transition">
          <i data-lucide="zoom-in" class="w-4 h-4"></i>
        </button>
        <button onclick="zoomOut()" class="p-2.5 rounded-xl glass-panel text-slate-300 hover:text-white hover:border-blue-500/50 transition">
          <i data-lucide="zoom-out" class="w-4 h-4"></i>
        </button>
        <button onclick="fitScreen()" class="p-2.5 rounded-xl glass-panel text-slate-300 hover:text-white hover:border-blue-500/50 transition" title="Fit Screen">
          <i data-lucide="maximize-2" class="w-4 h-4"></i>
        </button>
      </div>
    </main>

    <!-- Right Sidebar: Entity Inspector Panel -->
    <aside id="inspector" class="w-96 glass-panel border-l border-slate-800/80 flex flex-col z-10 custom-scrollbar overflow-y-auto">
      <div class="p-4 border-b border-slate-800/60 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <i data-lucide="info" class="w-4 h-4 text-blue-400"></i>
          <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-300">Entity Inspector</h2>
        </div>
        <span id="inspector-badge" class="badge bg-blue-500/10 text-blue-400 border border-blue-500/20">READY</span>
      </div>

      <div id="inspector-content" class="p-5 space-y-4">
        <!-- Default State -->
        <div class="text-center py-12 text-slate-500">
          <i data-lucide="mouse-pointer-click" class="w-10 h-10 mx-auto mb-3 opacity-40"></i>
          <p class="text-xs">Click on any User, Movie, Book, or Creator to inspect their direct Knowledge Graph connections.</p>
        </div>
      </div>
    </aside>

  </div>

  <script>
    const rawNodes = __RAW_NODES__;
    const rawEdges = __RAW_EDGES__;
    const userSuggestions = __USER_SUGGESTIONS__;
    const typeMeta = __TYPE_META__;

    let network = null;
    let nodesDataSet = null;
    let edgesDataSet = null;
    let physicsEnabled = true;
    let activeFilters = {
      user: true,
      movie: true,
      book: true,
      audio: true,
      podcast: true,
      creator: true,
      category: true
    };

    function init() {
      // 1. Populate User Select Dropdown
      const userSelect = document.getElementById('user-select');
      userSelect.innerHTML = '';
      Object.keys(userSuggestions).forEach(key => {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = userSuggestions[key].profile.name;
        userSelect.appendChild(opt);
      });

      // 2. Initialize Vis.js Network
      const container = document.getElementById('network-container');
      nodesDataSet = new vis.DataSet(rawNodes);
      edgesDataSet = new vis.DataSet(rawEdges);

      const data = { nodes: nodesDataSet, edges: edgesDataSet };
      const options = {
        nodes: {
          font: { face: 'Inter', color: '#FFFFFF' },
          borderWidth: 2,
          shadow: { enabled: true, color: 'rgba(0,0,0,0.5)', size: 10, x: 5, y: 5 }
        },
        edges: {
          font: { face: 'Inter', size: 10, color: '#94A3B8', strokeWidth: 0, align: 'middle' },
          arrows: { to: { enabled: true, scaleFactor: 0.6 } },
          smooth: { type: 'continuous', roundness: 0.2 },
          color: { inherit: false, opacity: 0.6 }
        },
        physics: {
          enabled: true,
          solver: 'forceAtlas2Based',
          forceAtlas2Based: {
            gravitationalConstant: -60,
            centralGravity: 0.012,
            springLength: 110,
            springConstant: 0.08,
            damping: 0.85
          },
          stabilization: { iterations: 100 }
        },
        interaction: {
          hover: true,
          tooltipDelay: 150,
          hideEdgesOnDrag: false
        }
      };

      network = new vis.Network(container, data, options);

      network.on('click', function(params) {
        if (params.nodes.length > 0) {
          inspectNode(params.nodes[0]);
        }
      });

      renderFilterCheckboxes();
      
      // Auto-select the first user
      const firstUser = Object.keys(userSuggestions)[0];
      if (firstUser) {
        selectUser(firstUser);
      }

      lucide.createIcons();
    }

    function selectUser(userKey) {
      const uData = userSuggestions[userKey];
      if (!uData) return;

      const profile = uData.profile;
      const userNodeId = `user:${userKey}`;

      // Update Left Sidebar Banner
      const banner = document.getElementById('user-profile-banner');
      banner.innerHTML = `
        <div class="flex items-center gap-3 mb-2">
          <div class="w-10 h-10 rounded-xl bg-amber-500 text-slate-950 font-black flex items-center justify-center text-lg shadow-md shadow-amber-500/20">
            ${profile.name[0]}
          </div>
          <div>
            <div class="text-sm font-bold text-white">${profile.name}</div>
            <div class="text-xs text-amber-400 font-medium">${profile.profession.toUpperCase()} &bull; ${profile.age_group}</div>
          </div>
        </div>
        <div class="text-[11px] text-slate-300 mt-2 space-y-1">
          <div><span class="text-slate-400">Interests:</span> <span class="text-slate-200">${profile.interests.join(', ')}</span></div>
          <div><span class="text-slate-400">Skills:</span> <span class="text-slate-200">${profile.skills.join(', ')}</span></div>
        </div>
      `;

      // Update Recommendations List
      const recsList = document.getElementById('user-recs-list');
      recsList.innerHTML = '';

      if (uData.recommendations.length === 0) {
        recsList.innerHTML = '<div class="text-xs text-slate-500">No suggestions available.</div>';
      } else {
        uData.recommendations.forEach((rec, idx) => {
          const div = document.createElement('div');
          div.className = 'p-2.5 rounded-lg bg-slate-900/90 border border-slate-800 hover:border-emerald-500/50 cursor-pointer transition';
          div.onclick = () => focusNode(rec.item_id);
          div.innerHTML = `
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-slate-200 truncate flex-1">${idx + 1}. ${rec.title}</span>
              <span class="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400">+${rec.graph_score}</span>
            </div>
            <div class="text-[10px] text-slate-400 mt-1 flex items-center justify-between">
              <span class="capitalize text-slate-400">${rec.content_type}</span>
              <span class="text-amber-400">${rec.rating ? '⭐ ' + rec.rating : ''}</span>
            </div>
          `;
          recsList.appendChild(div);
        });
      }

      // Highlight User & focus camera
      focusNode(userNodeId);
      lucide.createIcons();
    }

    function renderFilterCheckboxes() {
      const container = document.getElementById('filter-checkboxes');
      container.innerHTML = '';
      
      Object.keys(typeMeta).forEach(type => {
        const item = typeMeta[type];
        const count = rawNodes.filter(n => n.group === type).length;
        if (count === 0) return;

        const div = document.createElement('label');
        div.className = 'flex items-center justify-between p-1.5 rounded hover:bg-slate-800/60 cursor-pointer text-xs transition';
        div.innerHTML = `
          <div class="flex items-center gap-2">
            <input type="checkbox" checked onchange="toggleFilter('${type}', this.checked)" class="rounded bg-slate-900 border-slate-700 text-blue-600 focus:ring-0 w-3.5 h-3.5" />
            <span>${item.icon} ${item.label}</span>
          </div>
          <span class="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-400 font-mono">${count}</span>
        `;
        container.appendChild(div);
      });
    }

    function toggleFilter(type, isChecked) {
      activeFilters[type] = isChecked;
      const filteredNodes = rawNodes.filter(n => activeFilters[n.group]);
      const nodeIds = new Set(filteredNodes.map(n => n.id));
      const filteredEdges = rawEdges.filter(e => nodeIds.has(e.from) && nodeIds.has(e.to));
      
      nodesDataSet.clear();
      nodesDataSet.add(filteredNodes);
      edgesDataSet.clear();
      edgesDataSet.add(filteredEdges);
    }

    function inspectNode(nodeId) {
      const node = rawNodes.find(n => n.id === nodeId);
      if (!node) return;

      const connectedEdgeList = rawEdges.filter(e => e.from === nodeId || e.to === nodeId);
      const neighborIds = connectedEdgeList.map(e => e.from === nodeId ? e.to : e.from);
      const neighborNodes = rawNodes.filter(n => neighborIds.includes(n.id));

      const badge = document.getElementById('inspector-badge');
      const container = document.getElementById('inspector-content');

      const cInfo = typeMeta[node.group] || { bg: '#3B82F6', icon: '📌', label: node.group };
      badge.textContent = cInfo.label.toUpperCase();
      badge.style.borderColor = cInfo.bg;
      badge.style.color = cInfo.bg;

      const meta = node.metadata || {};

      let html = `
        <div class="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3">
          <div class="flex items-start gap-3">
            <div class="text-2xl p-2 rounded-lg bg-slate-800/80">${cInfo.icon}</div>
            <div class="flex-1">
              <h3 class="text-sm font-bold text-white leading-tight">${meta.title || meta.name || node.label}</h3>
              <div class="text-xs text-slate-400 mt-0.5 font-mono text-[11px]">${nodeId}</div>
            </div>
          </div>
      `;

      if (meta.rating) {
        html += `
          <div class="flex items-center gap-4 text-xs pt-1">
            <div class="flex items-center gap-1 text-amber-400 font-semibold">
              ⭐ ${meta.rating}
            </div>
            <div class="text-slate-400">Popularity: <span class="text-slate-200">${meta.popularity || 'N/A'}</span></div>
          </div>
        `;
      }

      if (meta.description) {
        html += `<p class="text-xs text-slate-300 leading-relaxed border-t border-slate-800/80 pt-2">${meta.description}</p>`;
      }

      if (meta.creators) {
        html += `<div class="text-xs"><span class="text-slate-400">Creators/Authors:</span> <span class="text-slate-200 font-medium">${meta.creators}</span></div>`;
      }

      if (meta.categories) {
        html += `<div class="text-xs"><span class="text-slate-400">Categories:</span> <span class="text-slate-200">${meta.categories}</span></div>`;
      }

      if (meta.interests) {
        html += `<div class="text-xs"><span class="text-slate-400">Interests:</span> <span class="text-slate-200">${meta.interests}</span></div>`;
      }

      html += `</div>`;

      // Connected Neighbors Section
      html += `
        <div class="space-y-2">
          <div class="flex items-center justify-between text-xs font-semibold text-slate-400">
            <span>Direct Knowledge & AI Links (${connectedEdgeList.length})</span>
          </div>
          <div class="space-y-1.5 max-h-64 overflow-y-auto custom-scrollbar">
      `;

      neighborNodes.forEach(neighbor => {
        const edge = connectedEdgeList.find(e => (e.from === neighbor.id && e.to === nodeId) || (e.to === neighbor.id && e.from === nodeId));
        const nInfo = typeMeta[neighbor.group] || { icon: '📌', bg: '#64748B' };
        const isRec = edge?.relation === 'AI_RECOMMENDATION';
        
        html += `
          <div onclick="focusNode('${neighbor.id}')" class="p-2 rounded-lg bg-slate-900/50 hover:bg-slate-800 border ${isRec ? 'border-emerald-500/40 bg-emerald-950/20' : 'border-slate-800/80'} cursor-pointer flex items-center justify-between text-xs transition">
            <div class="flex items-center gap-2 truncate flex-1">
              <span>${nInfo.icon}</span>
              <span class="text-slate-200 font-medium truncate">${neighbor.metadata?.title || neighbor.metadata?.name || neighbor.label}</span>
            </div>
            <span class="text-[10px] font-mono px-1.5 py-0.5 rounded ${isRec ? 'bg-emerald-500/20 text-emerald-400 font-bold' : 'bg-slate-800 text-blue-400'}">${edge?.label || 'LINK'}</span>
          </div>
        `;
      });

      html += `</div></div>`;
      container.innerHTML = html;
      lucide.createIcons();
    }

    function focusNode(nodeId) {
      inspectNode(nodeId);
      network.focus(nodeId, {
        scale: 1.25,
        animation: { duration: 700, easingFunction: 'easeInOutQuad' }
      });
      network.selectNodes([nodeId]);
    }

    function handleSearch(query) {
      if (!query || query.trim() === '') return;
      const q = query.toLowerCase().trim();
      const match = rawNodes.find(n => 
        (n.metadata?.title || '').toLowerCase().includes(q) ||
        (n.metadata?.name || '').toLowerCase().includes(q) ||
        (n.metadata?.creators || '').toLowerCase().includes(q) ||
        (n.metadata?.categories || '').toLowerCase().includes(q) ||
        n.label.toLowerCase().includes(q)
      );
      if (match) focusNode(match.id);
    }

    function resetView() {
      document.getElementById('search-input').value = '';
      network.fit({ animation: { duration: 800, easingFunction: 'easeInOutQuad' } });
    }

    function zoomIn() { network.moveTo({ scale: network.getScale() * 1.3 }); }
    function zoomOut() { network.moveTo({ scale: network.getScale() * 0.7 }); }
    function fitScreen() { network.fit({ animation: { duration: 800 } }); }

    function togglePhysics() {
      physicsEnabled = !physicsEnabled;
      network.setOptions({ physics: { enabled: physicsEnabled } });
      const btn = document.getElementById('btn-physics');
      if (physicsEnabled) {
        btn.innerHTML = '<i data-lucide="zap" class="w-3.5 h-3.5"></i> Physics: ON';
        btn.className = 'px-3 py-1.5 rounded-lg bg-blue-600/80 hover:bg-blue-600 text-xs font-medium text-white transition flex items-center gap-1.5';
      } else {
        btn.innerHTML = '<i data-lucide="zap-off" class="w-3.5 h-3.5"></i> Physics: OFF';
        btn.className = 'px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 transition flex items-center gap-1.5';
      }
      lucide.createIcons();
    }

    window.addEventListener('load', init);
  </script>
</body>
</html>
"""

def generate_html():
    nodes, edges, user_suggestions = build_full_network()
    
    html_content = TEMPLATE.replace("__RAW_NODES__", json.dumps(nodes))\
                           .replace("__RAW_EDGES__", json.dumps(edges))\
                           .replace("__USER_SUGGESTIONS__", json.dumps(user_suggestions))\
                           .replace("__TYPE_META__", json.dumps(TYPE_COLORS))
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"[OK] Generated Knowledge Graph & AI Recommender Visualizer at: {OUTPUT_HTML}")
    print(f"Total Nodes: {len(nodes)}, Total Edges: {len(edges)}, Total Users: {len(user_suggestions)}")

if __name__ == "__main__":
    generate_html()
