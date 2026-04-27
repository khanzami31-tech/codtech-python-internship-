"""
============================================================
CODTECH INTERNSHIP - TASK 4
Machine Learning Model Implementation
Spam Email Detection using Scikit-Learn
Models: Naive Bayes, Logistic Regression, SVM
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, shutil
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report)

print("=" * 60)
print("  CODTECH INTERNSHIP – TASK 4: SPAM EMAIL DETECTION")
print("=" * 60)

# ─── 1. Dataset ───────────────────────────────────────────────
print("\n[1] Preparing dataset...")

spam = [
    "Congratulations! You've won a FREE iPhone. Click here to claim your prize!",
    "WINNER!! You have been selected to receive a £900 prize reward! Text WIN to 80085",
    "FREE entry in our competition to win FA Cup tickets. Text FA to 87121 to enter",
    "You have 1 new voicemail. Call +1-800-PRIZE to collect your reward now!",
    "URGENT: Your account has been compromised. Send your password immediately.",
    "Get rich quick! Make $5000 per week working from home. No experience needed!",
    "Buy cheap Viagra online! Best prices guaranteed. No prescription needed.",
    "You are selected for a cash prize of $1,000,000. Reply to claim your money.",
    "Hot singles in your area! Click here for free access tonight.",
    "Lose 30 pounds in 30 days! Miracle pill clinically proven. Limited stock!",
    "Your PayPal account is suspended. Verify at paypal-secure.xyz immediately.",
    "Earn $500 daily from home. Join thousands of successful members today!",
    "Limited time: 90% OFF all products. Buy now before this offer expires!",
    "You are pre-approved for a $50,000 loan. No credit check required!",
    "I am a Nigerian prince needing your help to transfer $10 million dollars.",
    "FREE ringtones! Text RING to 85069. Subscription charges £3/week apply.",
    "Claim your free $500 gift card. Survey takes only 2 minutes. Act now!",
    "Warning: Computer virus detected. Call Microsoft Support at 1-800-FAKE.",
    "Exclusive: Buy 1 get 5 FREE. Only for selected customers like you!",
    "You have been randomly chosen for a $1000 Amazon gift card. Confirm now!",
    "Investment opportunity: 200% returns guaranteed. Send $100 to start.",
    "Your subscription expired. Click to renew before account deletion.",
    "FREE casino chips! Register now and get $500 bonus. No deposit required.",
    "ALERT: Unusual activity detected on your bank account. Secure it now.",
    "Weight loss secret celebrities don't want you to know. Click to discover!",
    "You owe back taxes. Call IRS immediately at 1-800-FAKE or face arrest.",
    "Make money online! No skills required. Earn $1000 daily guaranteed!",
    "Congratulations you are our lucky winner! Claim holiday worth $5000.",
    "Special: Rolex watches at 95% discount. Very limited stock available!",
    "Your mortgage application approved! Call for £100,000 same-day cash.",
    "Double your bitcoin in 24 hours! Guaranteed returns. Send BTC now!",
    "You won a luxury vacation! Just pay the small processing fee of $99.",
    "Work from home earning $200/hour. No experience or skills needed.",
    "Cheap prescription drugs online. No doctor needed. Fast delivery.",
    "Your email was selected for a $10,000 prize. Reply with your details.",
]

ham = [
    "Hey, are you free for lunch tomorrow? Thinking about that new Italian place.",
    "Please find attached the meeting agenda for Thursday. Let me know your thoughts.",
    "Can you review the code I pushed to the main branch? It fixes the login bug.",
    "Happy birthday! Hope you have a wonderful day with your family.",
    "The quarterly report is due Friday. Can you share your section by Wednesday?",
    "Just wanted to check in — how are you settling into the new project?",
    "Reminder: team standup at 10am. Please update your Jira tickets beforehand.",
    "The library you mentioned, scikit-learn, has excellent documentation online.",
    "Can you pick up some groceries on your way home? We need milk and bread.",
    "I enjoyed our conversation yesterday. Looking forward to the next session.",
    "Your flight to Mumbai is confirmed. Boarding pass attached to this email.",
    "The client approved the designs! They want to move forward next Monday.",
    "Netflix recommendation: you should watch Oppenheimer if you haven't yet.",
    "Python 3.12 was released with significant performance improvements.",
    "Let's reschedule our 3pm call to 4pm — something came up on my end.",
    "The internship project is going well. All four tasks are almost complete.",
    "Mom called — she wants to know if you're coming home this weekend.",
    "Great job on the presentation today! The client was very impressed.",
    "I've attached the invoice for last month. Please process it at your earliest.",
    "The new coffee machine is installed in the kitchen. Highly recommend trying it.",
    "Your package has been dispatched and will arrive by Thursday afternoon.",
    "Can you explain how the TF-IDF vectorizer works in scikit-learn? I'm confused.",
    "The project deadline has been extended to the end of next month.",
    "I've uploaded the dataset to Google Drive. The link is in the Slack channel.",
    "Thanks for covering my shift last week. I owe you one!",
    "The wifi password at the office has been changed. New one is in the email.",
    "Looking forward to the team dinner on Friday. Should be a great evening.",
    "Could you send over the updated resume? HR needs it for their records.",
    "The bug you reported has been fixed in the latest release version.",
    "Let me know if you need any help understanding the machine learning concepts.",
    "Dinner tonight at 7? I know a great place near the train station.",
    "Your doctor's appointment is confirmed for Tuesday at 11am.",
    "The new intern starts on Monday — please help them get settled in.",
    "I reviewed your pull request and left a few minor comments. Looks good overall.",
    "Can we discuss the project requirements in detail during tomorrow's meeting?",
]

texts  = spam + ham
labels = [1]*len(spam) + [0]*len(ham)   # 1=spam, 0=ham
df     = pd.DataFrame({'text': texts, 'label': labels})
df     = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
print(f"   Dataset: {len(df)} emails ({sum(labels)} spam, {len(labels)-sum(labels)} ham)")

# ─── 2. Train/Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.25, random_state=42, stratify=df['label'])
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# ─── 3. Train Three Models ────────────────────────────────────
print("\n[2] Training models...")

models = {
    "Naive Bayes":        Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')), ('clf', MultinomialNB())]),
    "Logistic Regression":Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')), ('clf', LogisticRegression(max_iter=1000))]),
    "SVM (LinearSVC)":    Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')), ('clf', LinearSVC(max_iter=2000))]),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv     = cross_val_score(model, df['text'], df['label'], cv=5, scoring='f1').mean()
    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "cv_f1":     cv,
        "cm":        confusion_matrix(y_test, y_pred),
    }
    print(f"   ✅ {name}: Acc={results[name]['accuracy']:.3f}, F1={results[name]['f1']:.3f}, CV-F1={cv:.3f}")

# ─── 4. Visualization Dashboard ──────────────────────────────
print("\n[3] Building evaluation dashboard...")

COLORS = ['#E63946','#457B9D','#2A9D8F']
BG     = '#F8F9FA'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})

fig = plt.figure(figsize=(20, 20), facecolor=BG)
fig.suptitle('🤖  Spam Detection – ML Model Evaluation Dashboard\nCodTech Internship Task 4  •  scikit-learn',
    fontsize=18, fontweight='bold', y=0.99, color='#1D3557')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.40)

model_names  = list(results.keys())
short_names  = ["Naive\nBayes", "Logistic\nReg.", "SVM"]
metrics_list = ['accuracy','precision','recall','f1']

# Plot 1 – Grouped Metric Bars
ax1 = fig.add_subplot(gs[0, :2]); ax1.set_facecolor(BG)
x = np.arange(len(model_names)); w = 0.2
for j, metric in enumerate(metrics_list):
    vals = [results[n][metric] for n in model_names]
    bars = ax1.bar(x + j*w, vals, w, label=metric.capitalize(), color=['#E63946','#457B9D','#2A9D8F','#E9C46A'][j], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{v:.2f}',
                 ha='center', va='bottom', fontsize=7, fontweight='bold')
ax1.set_xticks(x + w*1.5); ax1.set_xticklabels(short_names)
ax1.set_title('📊  Model Performance Metrics', fontsize=13, fontweight='bold', color='#1D3557', pad=10)
ax1.set_ylim(0, 1.12); ax1.set_ylabel('Score')
ax1.legend(fontsize=9, loc='lower right'); ax1.grid(axis='y', alpha=0.3, zorder=0)

# Plot 2 – CV F1 Scores
ax2 = fig.add_subplot(gs[0, 2]); ax2.set_facecolor(BG)
cv_vals = [results[n]['cv_f1'] for n in model_names]
bars2   = ax2.bar(short_names, cv_vals, color=COLORS, edgecolor='white', zorder=3)
ax2.set_title('🔁  5-Fold Cross-Val F1', fontsize=13, fontweight='bold', color='#1D3557', pad=10)
ax2.set_ylim(0, 1.1); ax2.set_ylabel('F1 Score'); ax2.grid(axis='y', alpha=0.3, zorder=0)
for bar, v in zip(bars2, cv_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3,4,5 – Confusion Matrices
for idx, (name, color) in enumerate(zip(model_names, COLORS)):
    ax = fig.add_subplot(gs[1, idx])
    cm = results[name]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'],
                linewidths=2, linecolor='white', cbar=False, annot_kws={'size':14,'weight':'bold'})
    ax.set_title(f'🔲  {short_names[idx].replace(chr(10)," ")}\nConfusion Matrix',
                 fontsize=11, fontweight='bold', color='#1D3557', pad=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

# Plot 6 – Class Distribution
ax6 = fig.add_subplot(gs[2, 0]); ax6.set_facecolor(BG)
ax6.pie([sum(labels), len(labels)-sum(labels)], labels=['Spam','Ham'],
        colors=['#E63946','#457B9D'], autopct='%1.0f%%', startangle=90,
        wedgeprops={'edgecolor':'white','linewidth':2})
ax6.set_title('📂  Dataset Distribution', fontsize=13, fontweight='bold', color='#1D3557', pad=10)

# Plot 7 – Radar / Spider chart
ax7 = fig.add_subplot(gs[2, 1], polar=True); 
cats    = ['Accuracy','Precision','Recall','F1']
N       = len(cats)
angles  = [n/float(N)*2*np.pi for n in range(N)] + [0]
for i, (name, color) in enumerate(zip(model_names, COLORS)):
    vals_r = [results[name][m.lower()] for m in cats] + [results[name]['accuracy']]
    ax7.plot(angles, vals_r, 'o-', linewidth=2, color=color, label=short_names[i].replace('\n',' '))
    ax7.fill(angles, vals_r, alpha=0.08, color=color)
ax7.set_xticks(angles[:-1]); ax7.set_xticklabels(cats, fontsize=9)
ax7.set_ylim(0,1); ax7.set_title('🕸️  Radar: All Metrics', fontsize=11, fontweight='bold', color='#1D3557', pad=20)
ax7.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=8)

# Plot 8 – Live Predictions
ax8 = fig.add_subplot(gs[2, 2]); ax8.axis('off')
best_model_name = max(results, key=lambda n: results[n]['f1'])
best_model      = results[best_model_name]['model']

test_emails = [
    ("Congratulations! Win FREE iPhone now!", "SPAM"),
    ("Meeting at 10am tomorrow, see you then", "HAM"),
    ("Claim $1000 prize, click here fast!",   "SPAM"),
    ("Can you review my pull request please?", "HAM"),
    ("FREE Viagra, no prescription needed",   "SPAM"),
    ("Your invoice is attached, thank you",   "HAM"),
]

table_data = [["Email (truncated)", "Actual", "Predicted", "✓"]]
for email, actual in test_emails:
    pred    = "SPAM" if best_model.predict([email])[0] == 1 else "HAM"
    correct = "✅" if pred == actual else "❌"
    table_data.append([email[:32]+"...", actual, pred, correct])

tbl = ax8.table(cellText=table_data[1:], colLabels=table_data[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.8)
for (r,c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1D3557'); cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#EEF4FF')
    cell.set_edgecolor('white')
ax8.set_title(f'🧪  Live Predictions ({best_model_name})', fontsize=11, fontweight='bold', color='#1D3557', pad=10)

plt.savefig('/mnt/user-data/outputs/task4_ml_dashboard.png', dpi=140, bbox_inches='tight', facecolor=BG)
shutil.copy('/home/claude/task4_ml_model.py', '/mnt/user-data/outputs/task4_ml_model.py')
print("   ✅ Dashboard saved!")

# ─── 5. Summary ──────────────────────────────────────────────
print("\n── Model Results Summary ────────────────────────────────")
print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'CV-F1':>8}")
print("─"*70)
for name in model_names:
    r = results[name]
    print(f"{name:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['cv_f1']:>8.4f}")
print(f"\n🏆 Best Model: {max(results, key=lambda n: results[n]['f1'])}")
print("\n✅ Task 4 Complete!")
