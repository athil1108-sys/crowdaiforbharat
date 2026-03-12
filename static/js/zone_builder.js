/**
 * ZoneBuilder — Interactive canvas zone drawing tool.
 * Coordinates are stored in "canvas display pixel" space.
 * Caller is responsible for scaling to backend coordinate space before saving.
 */
class ZoneBuilder {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.image = new Image();
        this.zones = [];
        this.isDrawing = false;
        this.startX = 0;
        this.startY = 0;
        this.currentRect = null;
        this._initEvents();
    }

    loadImage(dataUrl) {
        this.image = new Image();
        this.image.onload = () => this.render();
        this.image.src = dataUrl;
    }

    _getPos(e) {
        const r = this.canvas.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top };
    }

    _initEvents() {
        this.canvas.addEventListener('mousedown', (e) => {
            const p = this._getPos(e);
            this.isDrawing = true;
            this.startX = p.x;
            this.startY = p.y;
        });
        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.isDrawing) return;
            const p = this._getPos(e);
            this.currentRect = {
                x: Math.min(p.x, this.startX),
                y: Math.min(p.y, this.startY),
                w: Math.abs(p.x - this.startX),
                h: Math.abs(p.y - this.startY)
            };
            this.render();
        });
        this.canvas.addEventListener('mouseup', () => {
            if (this.isDrawing && this.currentRect && this.currentRect.w > 10 && this.currentRect.h > 10) {
                const name = prompt(`Name this zone:`, `Zone_${String.fromCharCode(65 + this.zones.length)}`);
                if (name && name.trim()) {
                    this.zones.push({ ...this.currentRect, name: name.trim() });
                }
            }
            this.isDrawing = false;
            this.currentRect = null;
            this.render();
        });
        this.canvas.addEventListener('mouseleave', () => {
            if (this.isDrawing) {
                this.isDrawing = false;
                this.currentRect = null;
                this.render();
            }
        });
    }

    render() {
        const { width: cw, height: ch } = this.canvas;
        this.ctx.clearRect(0, 0, cw, ch);

        // Draw blueprint background
        if (this.image.src) {
            try { this.ctx.drawImage(this.image, 0, 0, cw, ch); } catch (_) {}
        } else {
            this.ctx.fillStyle = '#07111E';
            this.ctx.fillRect(0, 0, cw, ch);
        }

        const palette = ['#22D3EE','#818CF8','#34D399','#F59E0B','#c084fc','#fb923c'];

        // Draw saved zones
        this.zones.forEach((z, i) => {
            const col = palette[i % palette.length];
            this.ctx.strokeStyle = col;
            this.ctx.lineWidth = 2;
            this.ctx.fillStyle = col + '22';
            this.ctx.fillRect(z.x, z.y, z.w, z.h);
            this.ctx.strokeRect(z.x, z.y, z.w, z.h);
            this.ctx.fillStyle = col;
            this.ctx.font = 'bold 13px JetBrains Mono, monospace';
            this.ctx.fillText(z.name, z.x + 8, z.y + 20);
        });

        // Draw active drag rect
        if (this.currentRect) {
            this.ctx.strokeStyle = '#F59E0B';
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([6, 4]);
            this.ctx.fillStyle = 'rgba(245,158,11,0.1)';
            this.ctx.fillRect(this.currentRect.x, this.currentRect.y, this.currentRect.w, this.currentRect.h);
            this.ctx.strokeRect(this.currentRect.x, this.currentRect.y, this.currentRect.w, this.currentRect.h);
            this.ctx.setLineDash([]);
        }
    }

    /** Returns zones scaled to the canonical 900×520 backend coordinate space */
    getScaledZones() {
        const scaleX = 900 / this.canvas.clientWidth;
        const scaleY = 520 / this.canvas.clientHeight;
        return this.zones.map(z => ({
            name: z.name,
            x: Math.round(z.x * scaleX),
            y: Math.round(z.y * scaleY),
            w: Math.round(z.w * scaleX),
            h: Math.round(z.h * scaleY)
        }));
    }

    removeZone(index) {
        this.zones.splice(index, 1);
        this.render();
    }

    clear() {
        this.zones = [];
        this.render();
    }
}
