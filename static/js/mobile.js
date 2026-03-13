document.addEventListener('DOMContentLoaded', () => {
    const app = {
        views: {
            login: document.getElementById('login-view'),
            hub: document.getElementById('hub-view'),
            detail: document.getElementById('detail-view'),
            profile: document.getElementById('profile-view')
        },
        activeZoneId: null,
        pollingInterval: null,
        activeTab: 'alerts',
        notifications: [],

        init() {
            this.bindEvents();
            this.switchView('login');
        },

        bindEvents() {
            // Auth Button
            document.getElementById('btn-authenticate').addEventListener('click', () => {
                this.switchTab('alerts');
                this.startPolling();
            });

            // Nav Items
            document.querySelectorAll('.nav-item').forEach(item => {
                item.addEventListener('click', () => {
                    const tab = item.getAttribute('data-tab');
                    this.switchTab(tab);
                });
            });

            // Back Buttons
            document.querySelectorAll('.btn-back, .btn-back-to-hub, .btn-return').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.switchTab('alerts');
                });
            });

            // Notifications Bell
            const bell = document.getElementById('btn-notifications');
            if (bell) {
                bell.onclick = () => this.showNotificationsModal();
            }

            // Close Modal
            const closeBtn = document.querySelector('.btn-close-modal');
            if (closeBtn) {
                closeBtn.onclick = () => this.hideNotificationsModal();
            }

            // Also close modal on click outside content
            const modal = document.getElementById('notifications-modal');
            if (modal) {
                modal.onclick = (e) => {
                    if (e.target === modal) this.hideNotificationsModal();
                };
            }

            // Logout
            document.querySelector('.btn-logout').addEventListener('click', () => {
                clearInterval(this.pollingInterval);
                this.switchView('login');
            });
        },

        switchTab(tabName) {
            this.activeTab = tabName;
            
            // Update Nav UI
            document.querySelectorAll('.nav-item').forEach(i => {
                i.classList.toggle('active', i.getAttribute('data-tab') === tabName);
            });

            if (tabName === 'alerts') this.switchView('hub');
            else if (tabName === 'zones') this.switchView('hub'); // In this consolidated prototype, Zones/Alerts are the Hub
            else if (tabName === 'profile') this.switchView('profile');
        },

        switchView(viewName) {
            Object.values(this.views).forEach(v => {
                if(v) v.classList.remove('active');
            });
            if (this.views[viewName]) {
                this.views[viewName].classList.add('active');
            }

            // Global Nav Visibility
            const globalNav = document.getElementById('main-nav');
            if (globalNav) {
                if (viewName === 'hub' || viewName === 'profile') {
                    globalNav.classList.remove('hidden');
                } else {
                    globalNav.classList.add('hidden');
                }
            }
        },

        async startPolling() {
            this.updateData();
            this.pollingInterval = setInterval(() => this.updateData(), 3000);
        },

        async updateData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                
                const zoneArray = data.zones ? 
                    Object.entries(data.zones).map(([key, val]) => ({
                        ...val,
                        name: key.replace(/_/g, ' ')
                    })) : [];
                
                this.renderZoneList(zoneArray, 'zone-list');
                this.fetchAIOverview();
                
                if (this.activeZoneId) {
                    const zone = zoneArray.find(z => z.name === this.activeZoneId);
                    if (zone) this.updateDetailView(zone);
                }
            } catch (e) {
                console.error("Scale polling failed", e);
            }
        },

        async fetchAIOverview() {
            try {
                const res = await fetch('/api/overview');
                const data = await res.json();
                const descEl = document.getElementById('ai-command-desc');
                
                if (data.overview && descEl) {
                    // Local shortening for mobile
                    let text = data.overview;
                    // Remove markdown bolding
                    text = text.replace(/\*\*/g, '');
                    // Take first sentence or first 120 chars
                    let shortText = text.split('.')[0] + '.';
                    if (shortText.length > 120) shortText = text.substring(0, 117) + '...';
                    
                    if (descEl.innerText !== shortText) {
                        descEl.innerText = shortText;
                        this.addNotification('AI Command', shortText);
                    }
                }
            } catch (e) {
                console.error("AI Overview fetch failed", e);
            }
        },

        renderZoneList(zones, containerId) {
            const container = document.getElementById(containerId);
            if (!container) return;
            container.innerHTML = '';

            zones.forEach(zone => {
                const zoneName = zone.name || zone.zone_id || "Unknown Zone";
                const risk = zone.risk_level || "green";
                
                const card = document.createElement('div');
                card.className = `zone-card ${risk}`;
                
                let iconClass = 'fa-users';
                if (zoneName.includes('Parking')) iconClass = 'fa-square-p';
                if (zoneName.includes('Stage')) iconClass = 'fa-circle-exclamation';

                const description = zone.message || (risk === 'red' ? 'CRITICAL: Capacity Reached' : 'Monitoring crowd density');

                card.innerHTML = `
                    <div class="zone-icon">
                        <i class="fa-solid ${risk === 'red' ? 'fa-triangle-exclamation' : iconClass}"></i>
                    </div>
                    <div class="zone-info">
                        <h4>${zoneName}</h4>
                        <p>${description}</p>
                    </div>
                    <div class="zone-status-tag status-${risk}">
                        ${risk.toUpperCase()}
                    </div>
                    <i class="fa-solid fa-chevron-right" style="color: #444; font-size: 12px;"></i>
                `;

                card.addEventListener('click', () => {
                    this.activeZoneId = zoneName;
                    this.switchView('detail');
                    this.updateDetailView(zone);
                });

                container.appendChild(card);
            });

            const activeCountEl = document.querySelector('.active-count');
            if (activeCountEl) activeCountEl.innerText = `${zones.length} Active Zones`;
        },

        updateDetailView(zone) {
            const zoneName = zone.name || zone.zone_id || "Unknown Zone";
            document.getElementById('detail-zone-name').innerText = zoneName;
            
            const densityVal = document.getElementById('detail-density');
            const flowVal = document.getElementById('detail-flow');
            
            if (densityVal) densityVal.innerText = `${Math.round(parseFloat(zone.density) * 10)}%`;
            if (flowVal) flowVal.innerHTML = `${Math.round(parseFloat(zone.velocity) * 150)} <small>ppm</small>`;
            
            // Render notifications matching the image
            const notifList = document.getElementById('zone-notification-list');
            if (notifList) {
                notifList.innerHTML = `
                    <div class="notification-item">
                        <div class="notif-dot orange"></div>
                        <div class="notif-content">
                            <div class="notif-header">
                                <h5>High Density Warning</h5>
                                <span class="notif-time">2m ago</span>
                            </div>
                            <p class="notif-desc">${zoneName} entrance bottlenecking. Recommend redirecting flow to West Gates.</p>
                        </div>
                    </div>
                    <div class="notification-item">
                        <div class="notif-dot grey"></div>
                        <div class="notif-content">
                            <div class="notif-header">
                                <h5>Staff Deployment</h5>
                                <span class="notif-time">15m ago</span>
                            </div>
                            <p class="notif-desc">Response Team 4 reaching ${zoneName} checkpoint.</p>
                        </div>
                    </div>
                    <div class="notification-item">
                        <div class="notif-dot red"></div>
                        <div class="notif-content">
                            <div class="notif-header">
                                <h5>Capacity Alert</h5>
                                <span class="notif-time">45m ago</span>
                            </div>
                            <p class="notif-desc">${zoneName} mosh pit area reached 95% capacity threshold.</p>
                        </div>
                    </div>
                `;
            }

            // Also add a real notification to the global list if risk is high
            const risk = zone.risk_level || "green";
            if (risk === 'red' || risk === 'yellow') {
                this.addNotification(`${zoneName} Alert`, zone.message || `Critical density at ${zoneName}.`);
            }
        },

        addNotification(title, msg) {
            const id = Date.now();
            // Avoid duplicates of the same message within short window
            if (this.notifications.some(n => n.msg === msg)) return;
            
            this.notifications.unshift({ id, title, msg, time: 'Just now' });
            if (this.notifications.length > 20) this.notifications.pop();
            
            this.updateNotificationBadge();
        },

        updateNotificationBadge() {
            const badge = document.querySelector('.bell-box .badge');
            if (badge) {
                if (this.notifications.length > 0) {
                    badge.style.display = 'block';
                } else {
                    badge.style.display = 'none';
                }
            }
        },

        showNotificationsModal() {
            const modal = document.getElementById('notifications-modal');
            const list = document.getElementById('modal-notification-list');
            
            if (modal && list) {
                list.innerHTML = this.notifications.map(n => `
                    <div class="notification-item">
                        <div class="notif-dot"></div>
                        <div class="notif-body">
                            <h5>${n.title}</h5>
                            <p>${n.msg}</p>
                            <span class="notif-time">${n.time}</span>
                        </div>
                    </div>
                `).join('');
                
                modal.classList.add('active');
                // Optional: clear badge when viewing
                const badge = document.querySelector('.bell-box .badge');
                if (badge) badge.style.display = 'none';
            }
        },

        hideNotificationsModal() {
            const modal = document.getElementById('notifications-modal');
            if (modal) modal.classList.remove('active');
        }
    };

    app.init();
});
