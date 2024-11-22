// Javascript library for displaying and manipulating the hexz board
// as well as communicating with the server.

const URL_PREFIX = document.querySelector('meta[name="hexz-url-prefix"]')?.content || "";

const styles = {
    colors: {
        grid: '#cbcbcb',
        players: ['#255ab4', '#f8d748'],
        hiddenMoves: ['#92acd9', '#fbeba3'], // 50% alpha of players on white bg.
        // blocked: ['rgba(248, 215, 72, 0.2)', 'rgba(37, 90, 180, 0.5)', '#3e3e3e'],  // same as hiddenMoves / unavailablePiece.
        blocked: ['#fbeba3', '#92acd9', '#5f5f5f'],  // same as hiddenMoves / rockCell.
        cellIcons: ['#1e1e1e', '#7a6505'], // Blue: background. Yellow: 25% HSL luminance of player's color.
        deadCell: '#d03d12',
        grassCell: '#008048',
        rockCell: '#5f5f5f',
        grassCellFg: '#00331d',
        unavailablePiece: '#3e3e3e',
        unavailablePieceIcon: '#cbcbcb',
    },
};

// Known game types. Keep in sync with GameType constants used by server.
const gameTypeFlagz = "Flagz";
const gameTypeClassic = "Classic";
const gameTypeFreeform = "Freeform";

// Cell types. Keep in sync with CellType constants used by server.
const cellNormal = 0;
const cellDead = cellNormal + 1;
const cellGrass = cellDead + 1;
const cellRock = cellGrass + 1;
const cellFire = cellRock + 1;
const cellFlag = cellFire + 1;
const cellPest = cellFlag + 1;
const cellDeath = cellPest + 1;


// Represents the game state.
const gstate = {
    board: null,
    role: 0,
    done: false,
    selectedCellType: 0,
    playerNames: [],
    // Optional information about the scores a CPU assigns to each move.
    moveScores: null,
    renderMoveScores: false,
    // Whether the CPU player is hosted client-side (as a WASM worker).
    clientSideCPUPlayer: false,
};

// Gets dynamically updated depending on the canvas size.
var hexagonSideLength = 30;

function gameId() {
    let pathSegs = window.location.pathname.split("/");
    return pathSegs[pathSegs.length - 1];
}

async function sendMove(row, col, cellType) {
    return fetch(`${URL_PREFIX}/move/${gameId()}`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            move: gstate.board.move,
            row: row,
            col: col,
            type: cellType,
        }),
    })
}

async function resetGame() {
    return fetch(`${URL_PREFIX}/reset/${gameId()}`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            message: "reset",
        }),
    });
}

function undoRedoMove(command) {
    return async function () {
        if (!gstate.board) {
            return
        }
        return fetch(`${URL_PREFIX}/${command}/${gameId()}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                move: gstate.board.move,
            }),
        });
    };
}

function handleServerEvent(sse, serverEvent) {
    if (serverEvent.role > 0) {
        gstate.role = serverEvent.role;
    }
    if (serverEvent.disableUndo) {
        console.log("Disabling undo/redo menu");
        let menurow = document.getElementById("menurow-undo-redo");
        menurow.style.display = "none";
    }
    if (serverEvent?.gameInfo?.clientSideCPUPlayer) {
        gstate.clientSideCPUPlayer = true;
        startWASMWebWorker();
    }
    if (serverEvent.board != null) {
        // new board received.
        gstate.board = serverEvent.board;
        if (serverEvent.gameInfo) {
            initializeButtonCells(serverEvent.gameInfo);
        }
        redraw();
        updateTurnInfo();
        updateScore();
        if (gstate.clientSideCPUPlayer && gstate.role == 1 && gstate.board.turn == 2) {
            // We are P1, it's CPU's turn. Wait 100ms to let the current redraw finish.
            sendWASMWorkerMoveRequest();
        }
        if (gstate.role > 0 && serverEvent.winner > 0) {
            // Show an animation if a winner was just announced.
            setTimeout(function () {
                showAnimation(serverEvent.winner == gstate.role ? animateWinner : animateLoser);
            }, 1000);
        }
    }
    if (serverEvent.playerNames) {
        gstate.playerNames = serverEvent.playerNames;
        updatePlayerNames();
    }
    if (serverEvent.debugMessage.length > 0) {
        console.log(serverEvent.timestamp + ": " + serverEvent.debugMessage);
    }
    if (serverEvent.lastEvent) {
        console.log("Server sent last SSE. Closing the connection.");
        gstate.done = true;
        sse.close();
    }
    if (serverEvent.announcements && serverEvent.announcements.length > 0) {
        updateAnnouncements(serverEvent);
    }
}

function updateAnnouncements(serverEvent) {
    let f2 = function (n) {
        return String(n).padStart(2, '0');
    }
    let div = document.getElementById("announcements");
    let time = new Date(serverEvent.timestamp);
    let timeStr = `${f2(time.getHours())}:${f2(time.getMinutes())}:${f2(time.getSeconds())}`;
    let text = [];
    for (const a of serverEvent.announcements) {
        text.push(`${timeStr} - ${a}`);
    }
    div.innerHTML = text.join("<br>");
}

function updateScore() {
    let div = document.getElementById("scoreInfo");
    const s = gstate.board.score;
    if (s) {
        div.innerHTML = s.join(" &ndash; ");
    }
}

function updateTurnInfo() {
    let ts = [
        document.getElementById("playerOneTurnInfo"),
        document.getElementById("playerTwoTurnInfo"),
    ];
    let turn = gstate.board.turn - 1;
    ts[turn].style.visibility = 'visible';
    ts[(turn + 1) % 2].style.visibility = 'hidden';
}

function updatePlayerNames() {
    let ps = [
        document.getElementById("playerOneBadge"),
        document.getElementById("playerTwoBadge"),
    ];
    for (let i = 0; i < gstate.playerNames.length; i++) {
        ps[i].innerHTML = gstate.playerNames[i];
    }
}

function newGame() {
    window.location.replace(`${URL_PREFIX}`);
}

// Returns a Path2D representing a 0-centered hexagon with side length a.
function hexagon(a) {
    let p = new Path2D();
    let b2 = Math.sqrt(3) * a / 2;
    let a2 = a / 2
    p.moveTo(b2, a2);
    p.lineTo(0, a);
    p.lineTo(-b2, a2);
    p.lineTo(-b2, -a2);
    p.lineTo(0, -a);
    p.lineTo(b2, -a2);
    p.lineTo(b2, a2);
    return p;
}

// Created with https://yqnn.github.io/svg-path-editor/, inspired by
// https://www.svgrepo.com/collection/zest-interface-icons
const cellTypePaths = {
    fire: "M -0.8562 -49.9998 L 3.7186 -44.2813 C 9.1362 -37.509 11.1328 -30.8269 10.9476 -24.416 C 10.7676 -18.1826 8.5309 -12.6255 6.2976 -8.0391 C 5.5471 -6.4981 4.7362 -4.95 3.9933 -3.5319 C 3.649 -2.8743 3.319 -2.2448 3.0167 -1.6567 C 2.0067 0.3076 1.2348 1.9333 0.7443 3.3595 C 0.2514 4.7914 0.16 5.6971 0.2229 6.2695 C 0.27 6.7014 0.4152 7.1362 0.9862 7.7071 C 1.9305 8.6514 2.6271 8.9167 3.0495 8.9976 C 3.4809 9.0809 4.0548 9.0538 4.8757 8.7167 C 6.7119 7.9614 8.9538 5.9952 11.3062 3.0457 C 13.5657 0.2124 15.5386 -3.0286 16.9686 -5.6219 C 17.6762 -6.9057 18.2357 -8.0025 18.6147 -8.7717 C 18.8038 -9.1558 18.9476 -9.4567 19.0419 -9.6566 L 19.1452 -9.8781 L 19.1676 -9.9277 L 19.1709 -9.9341 L 19.1714 -9.9357 L 19.1719 -9.9364 L 19.1719 -9.9366 L 21.9605 -16.1081 L 26.8247 -11.3921 C 37.8381 -0.7143 41.2461 14.7043 34.0071 28.7024 C 27.7119 40.8757 14.8205 49.1695 0 49.1695 C -20.9286 49.1695 -38.0952 32.5847 -38.0952 11.8824 C -38.0952 0.9757 -32.0625 -6.8286 -25.4918 -14.5012 C -24.6346 -15.5022 -23.7585 -16.5109 -22.8687 -17.5355 C -16.7675 -24.5605 -10.0167 -32.3334 -4.2276 -43.4985 L -0.8562 -49.9998 Z M 24.4138 0.5619 C 22.9276 3.1319 20.9947 6.1719 18.7519 8.9838 C 16.1195 12.2847 12.6324 15.8243 8.4981 17.5247 C 6.3333 18.4147 3.8614 18.8528 1.2495 18.35 C -1.3719 17.8452 -3.7162 16.4733 -5.7481 14.4414 C -7.8014 12.3881 -8.9543 9.9619 -9.2448 7.3038 C -9.52 4.7857 -8.9876 2.371 -8.2614 0.26 C -7.5328 -1.8562 -6.4895 -3.9957 -5.4529 -6.0114 C -5.0824 -6.7324 -4.7167 -7.4305 -4.3543 -8.1216 C -3.6486 -9.4677 -2.9562 -10.7892 -2.2652 -12.2084 C -0.21 -16.4289 1.3071 -20.5108 1.4276 -24.691 C 1.4924 -26.933 1.1581 -29.3329 0.13 -31.9123 C -5.1862 -23.3496 -10.8945 -16.7869 -15.7256 -11.2326 C -16.6011 -10.226 -17.4478 -9.2526 -18.2581 -8.3064 C -24.8819 -0.5719 -28.5714 4.8252 -28.5714 11.8824 C -28.5714 27.1062 -15.8904 39.6457 0 39.6457 C 11.2109 39.6457 20.8657 33.3819 25.5476 24.3276 C 29.6509 16.3928 29.0709 7.8014 24.4138 0.5619 Z",
    flag: "M 3.3072 -38.5457 C -5.0739 -45.2506 -14.9125 -45.0416 -21.9362 -43.5629 C -26.2188 -42.6613 -30.5708 -41.1876 -34.3913 -39.0083 C -36.0409 -38.0657 -37.0588 -36.3116 -37.0588 -34.4117 V 39.7059 C -37.0588 42.6298 -34.6885 45 -31.7647 45 C -28.8409 45 -26.4706 42.6298 -26.4706 39.7059 V 11.2733 C -19.185 8.2022 -9.9318 6.7759 -3.3072 12.0754 C 5.0739 18.7798 14.9125 18.5712 21.9362 17.0926 C 27.2959 15.9639 31.4688 14.0405 33.3286 13.1014 C 35.4012 12.0552 37.0588 10.4474 37.0588 7.9412 V -34.4117 C 37.0588 -36.299 36.054 -38.0434 34.4218 -38.9907 C 32.7907 -39.9373 30.7821 -39.9461 29.1441 -39.0116 L 29.1393 -39.0091 C 21.5264 -34.8494 10.5951 -32.7153 3.3072 -38.5457 Z M -26.4706 -31.0797 V -0.0053 C -16.5104 -3.1934 -5.1829 -2.9848 3.3072 3.807 C 8.1614 7.6908 14.2052 7.8999 19.755 6.7315 C 22.4132 6.1714 24.7601 5.3301 26.4706 4.6091 V -26.465 C 25.0888 -26.0227 23.5646 -25.6034 21.9362 -25.2606 C 14.9125 -23.7819 5.0739 -23.5729 -3.3072 -30.2777 C -9.9318 -35.5772 -19.185 -34.1507 -26.4706 -31.0797 Z",
    pest: "M -4.5455 -43.1818 C -4.5455 -39.4162 -7.5982 -36.3636 -11.3636 -36.3636 C -15.1292 -36.3636 -18.1818 -39.4162 -18.1818 -43.1818 C -18.1818 -46.9474 -15.1292 -50 -11.3636 -50 C -7.5982 -50 -4.5455 -46.9474 -4.5455 -43.1818 Z M -2.2727 -18.1818 C -2.2727 -15.0438 -4.8164 -12.5 -7.9545 -12.5 C -11.0927 -12.5 -13.6364 -15.0438 -13.6364 -18.1818 C -13.6364 -21.3198 -11.0927 -23.8636 -7.9545 -23.8636 C -4.8164 -23.8636 -2.2727 -21.3198 -2.2727 -18.1818 Z M 4.5455 -26.1364 C 7.6836 -26.1364 10.2273 -28.6802 10.2273 -31.8182 C 10.2273 -34.9562 7.6836 -37.5 4.5455 -37.5 C 1.4073 -37.5 -1.1364 -34.9562 -1.1364 -31.8182 C -1.1364 -28.6802 1.4073 -26.1364 4.5455 -26.1364 Z M 5.6818 18.1818 C 5.6818 21.32 3.1382 23.8636 0 23.8636 C -3.1382 23.8636 -5.6818 21.32 -5.6818 18.1818 C -5.6818 15.0436 -3.1382 12.5 0 12.5 C 3.1382 12.5 5.6818 15.0436 5.6818 18.1818 Z M -27.2727 -36.3636 C -29.7831 -36.3636 -31.8182 -34.3285 -31.8182 -31.8182 C -31.8182 -29.3078 -29.7831 -27.2727 -27.2727 -27.2727 V 22.7273 C -27.2727 28.5323 -25.5955 35.2159 -21.6166 40.5682 C -17.5166 46.0836 -11.0809 50 -2.2727 50 C 6.5355 50 12.9714 46.0836 17.0714 40.5682 C 21.05 35.2159 22.7273 28.5323 22.7273 22.7273 V -27.2727 C 25.2377 -27.2727 27.2727 -29.3078 27.2727 -31.8182 C 27.2727 -34.3285 25.2377 -36.3636 22.7273 -36.3636 H 18.1818 C 15.6714 -36.3636 13.6364 -34.3268 13.6364 -31.8165 V -2.295 C 11.6114 -1.0009 9.2532 0 6.8182 0 C 4.6114 0 2.3659 -0.8223 -0.9341 -2.0814 C -3.7286 -3.1482 -7.3914 -4.5455 -11.3636 -4.5455 C -13.8893 -4.5455 -16.2083 -3.9977 -18.1818 -3.27 L -18.1818 -31.8182 C -18.1818 -34.294 -20.2515 -36.3636 -22.7273 -36.3636 H -27.2727 Z M 6.8182 9.0909 C 9.3441 9.0909 11.6627 8.5432 13.6364 7.8155 V 22.7273 C 13.6364 26.9636 12.3782 31.6436 9.7755 35.1445 C 7.2941 38.4827 3.5023 40.9091 -2.2727 40.9091 C -8.0477 40.9091 -11.8395 38.4827 -14.3209 35.1445 C -16.9234 31.6436 -18.1818 26.9636 -18.1818 22.7273 L -18.1818 6.8405 C -16.1567 5.5464 -13.7985 4.5455 -11.3636 4.5455 C -9.1568 4.5455 -6.9114 5.3677 -3.6114 6.6268 C -0.8168 7.6936 2.8459 9.0909 6.8182 9.0909 Z",
    death: "M 33.3333 45.2381 C 33.3333 47.8681 31.2014 50 28.5714 50 H 9.5238 H -9.5238 H -28.5714 C -31.2014 50 -33.3333 47.8681 -33.3333 45.2381 V 34.5238 C -33.3333 29.9214 -37.0643 26.1905 -41.6667 26.1905 C -44.9541 26.1905 -47.619 23.5257 -47.619 20.2381 V -2.381 C -47.619 -28.6802 -26.2993 -50 0 -50 C 26.299 -50 47.619 -28.6802 47.619 -2.381 V 20.2381 C 47.619 23.5257 44.9543 26.1905 41.6667 26.1905 C 37.0643 26.1905 33.3333 29.9214 33.3333 34.5238 V 45.2381 Z M 23.8095 40.4762 V 34.5238 C 23.8095 25.8848 29.9443 18.6786 38.0952 17.0238 V -2.381 C 38.0952 -23.4204 21.0395 -40.4762 0 -40.4762 C -21.0394 -40.4762 -38.0952 -23.4204 -38.0952 -2.381 V 17.0238 C -29.9444 18.6786 -23.8095 25.8848 -23.8095 34.5238 V 40.4762 H -14.2857 V 30.9524 C -14.2857 28.3224 -12.1538 26.1905 -9.5238 26.1905 C -6.8938 26.1905 -4.7619 28.3224 -4.7619 30.9524 V 40.4762 H 4.7619 V 30.9524 C 4.7619 28.3224 6.8938 26.1905 9.5238 26.1905 C 12.1538 26.1905 14.2857 28.3224 14.2857 30.9524 V 40.4762 H 23.8095 Z M -4.7619 4.7619 C -4.7619 11.3367 -16.0368 16.6667 -22.6072 16.6667 C -28.6443 16.6667 -28.6092 12.1667 -28.5638 6.3362 V 6.3357 C -28.5598 5.821 -28.5557 5.2957 -28.5557 4.7619 C -28.5557 -1.8129 -23.2292 -7.1429 -16.6588 -7.1429 C -10.0883 -7.1429 -4.7619 -1.8129 -4.7619 4.7619 Z M 28.5638 6.3362 C 28.5595 5.821 28.5557 5.2957 28.5557 4.7619 C 28.5557 -1.8129 23.229 -7.1429 16.6586 -7.1429 C 10.0881 -7.1429 4.7619 -1.8129 4.7619 4.7619 C 4.7619 11.3367 16.0367 16.6667 22.6071 16.6667 C 28.6443 16.6667 28.609 12.1667 28.5638 6.3362 Z",
};

const cellTypeToPathMap = [];
cellTypeToPathMap[cellFire] = new Path2D(cellTypePaths.fire);
cellTypeToPathMap[cellFlag] = new Path2D(cellTypePaths.flag);
cellTypeToPathMap[cellPest] = new Path2D(cellTypePaths.pest);
cellTypeToPathMap[cellDeath] = new Path2D(cellTypePaths.death);

// All button cells the UI knows about, in the order they should be 
// displayed.
const allButtonCells = [
    {
        type: cellNormal,
    },
    {
        type: cellFire,
        symbol: cellTypeToPathMap[cellFire],
    },
    {
        type: cellFlag,
        symbol: cellTypeToPathMap[cellFlag],
    },
    {
        type: cellPest,
        symbol: cellTypeToPathMap[cellPest],
    },
    {
        type: cellDeath,
        symbol: cellTypeToPathMap[cellDeath],
    },
];
// Populated when the board arrives. Button cells that actually can 
// be used in the current game.
let buttonCells = [];

function initializeButtonCells(gameInfo) {
    if (!gstate.board || gstate.role == 0) {
        return;
    }
    let playerIdx = gstate.role - 1;
    buttonCells = [];
    for (let i = 0; i < allButtonCells.length; i++) {
        if (gameInfo.validCellTypes.includes(allButtonCells[i].type)) {
            buttonCells.push(allButtonCells[i]);
        }
    }
    if (gameInfo.gameType == gameTypeFlagz && gstate.board.move == 0) {
        gstate.selectedCellType = cellFlag;
    }
}

function drawBoard(ctx) {
    const board = gstate.board;
    let a = hexagonSideLength;
    let b = Math.sqrt(3) * a;
    let nRows = board.fields.length;
    let hex = hexagon(a);
    // Set properties of text drawn in cells.
    const fontSize = Math.floor(a);
    ctx.font = `${fontSize}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    // Draw cells.
    for (let i = 0; i < nRows; i++) {
        for (let j = 0; j < board.fields[i].length; j++) {
            let xOff = (i % 2 == 0) ? 0 : b / 2;
            let x = xOff + j * b + b / 2;
            let y = i * a * 3 / 2 + a;
            let fld = board.fields[i][j];
            ctx.translate(x, y);
            if (fld.owner) {
                if (fld.hidden) {
                    ctx.fillStyle = styles.colors.hiddenMoves[fld.owner - 1];
                    ctx.fill(hex);
                } else {
                    ctx.fillStyle = styles.colors.players[fld.owner - 1];
                    ctx.fill(hex);
                }
                if (fld.type != cellNormal) {
                    // Draw icon on top.
                    // TODO: factor this out, we use the same code for drawing button icons.
                    let symbol = cellTypeToPathMap[String(fld.type)];
                    if (symbol) {
                        ctx.save();
                        ctx.scale(b / 50 / 3.5, b / 50 / 3.5);
                        ctx.fillStyle = styles.colors.cellIcons[fld.owner - 1];
                        ctx.fill(symbol, 'evenodd');
                        ctx.restore();
                    }
                }
                if (fld.v > 0) {
                    ctx.fillStyle = styles.colors.cellIcons[fld.owner - 1];
                    ctx.fillText(String(fld.v), 0, 0);
                }
            } else if (fld.type == cellDead) {
                ctx.fillStyle = styles.colors.deadCell;
                ctx.fill(hex);
            } else if (fld.type == cellRock) {
                ctx.fillStyle = styles.colors.rockCell;
                ctx.fill(hex);
            } else if (fld.type == cellGrass) {
                ctx.fillStyle = styles.colors.grassCell;
                ctx.fill(hex);
                if (fld.v > 0) {
                    ctx.fillStyle = styles.colors.grassCellFg;
                    ctx.fillText(String(fld.v), 0, 0);
                }
            } else if (fld.blocked) {
                const blockedP1 = fld.blocked & 1;
                const blockedP2 = fld.blocked & 2;
                if (blockedP1) {
                    if (blockedP2) {
                        ctx.fillStyle = styles.colors.blocked[2];
                    } else {
                        ctx.fillStyle = styles.colors.blocked[0];
                    }
                    ctx.fill(hex);
                } else if (blockedP2) {
                    ctx.fillStyle = styles.colors.blocked[1];
                    ctx.fill(hex);
                }
            }
            if (!fld.owner && gstate.renderMoveScores && gstate.moveScores) {
                const n = gstate.moveScores.normalCell[i][j];
                const f = gstate.moveScores.flag[i][j];
                function cellValue(v) {
                    const oldFont = ctx.font;
                    const fontSize = Math.floor(0.4 * a);
                    ctx.font = `${fontSize}px sans-serif`;
                    ctx.fillStyle = "#303030";
                    ctx.fillText(v.toFixed(3), 0, 0);
                    ctx.font = oldFont;
                }
                if (n > f) {
                    //#800080
                    //#ffc0cb
                    ctx.fillStyle = scaleColor("#ffc0cb", "#800080", n);
                    ctx.fill(hex);
                    cellValue(n);
                } else if (f > n) {
                    ctx.fillStyle = scaleColor("#ffc0cb", "#800080", f);
                    ctx.fill(hex);
                    cellValue(f);
                } else if (n == f && n > 0) {
                    // Both scores are equal, let's choose placing a normal cell.
                    ctx.fillStyle = scaleColor("#ffc0cb", "#800080", n);
                    ctx.fill(hex);
                }
            }
            ctx.strokeStyle = styles.colors.grid;
            ctx.stroke(hex);
            // Undo transform.
            ctx.translate(-x, -y);
        }
    }
}

function scaleColor(startColor, endColor, scale) {
    const r1 = parseInt(startColor.slice(1, 3), 16);
    const g1 = parseInt(startColor.slice(3, 5), 16);
    const b1 = parseInt(startColor.slice(5, 7), 16);
    const r2 = parseInt(endColor.slice(1, 3), 16);
    const g2 = parseInt(endColor.slice(3, 5), 16);
    const b2 = parseInt(endColor.slice(5, 7), 16);
    const r = Math.round(r1 + (r2 - r1) * scale);
    const g = Math.round(g1 + (g2 - g1) * scale);
    const b = Math.round(b1 + (b2 - b1) * scale);
    return "#" + r.toString(16).padStart(2, "0")
        + g.toString(16).padStart(2, "0")
        + b.toString(16).padStart(2, "0");
}

function drawButtons(ctx) {
    if (!gstate.board || gstate.role == 0) {
        // Don't show buttons for spectators.
        return;
    }
    let playerIdx = gstate.role - 1;
    let a = hexagonSideLength;
    let b = Math.sqrt(3) * a;
    let nRows = gstate.board.fields.length;
    let hex = hexagon(a);
    // Draw bottom row with cell types to choose from.
    let bottomRow = nRows + 1;
    let xOff = (bottomRow % 2 == 0) ? 0 : b / 2;
    for (let j = 0; j < buttonCells.length; j++) {
        let x = xOff + (j * 1.3) * b + b / 2;
        let y = bottomRow * a * 3 / 2 + a;
        ctx.save();
        ctx.translate(x, y);
        // Determine background and icon colors to use.
        let available = buttonCells[j].type == cellNormal
            || !!gstate.board.resources[playerIdx].numPieces[buttonCells[j].type];
        let cellColor = styles.colors.hiddenMoves[playerIdx];
        if (!available) {
            cellColor = styles.colors.unavailablePiece;
        } else if (gstate.selectedCellType == buttonCells[j].type) {
            cellColor = styles.colors.players[playerIdx];
        }
        let iconColor = styles.colors.cellIcons[playerIdx];
        if (!available) {
            iconColor = styles.colors.unavailablePieceIcon;
        }
        ctx.fillStyle = cellColor;
        // Draw cell
        ctx.fill(hex);
        ctx.strokeStyle = styles.colors.grid;
        ctx.stroke(hex);
        // Draw symbol on top
        if (buttonCells[j].symbol) {
            // Every symbol is expected to be bounded by a ((-50, -50), (50, 50)) square.
            // 3.5 was chosen manually, icons "look right" at that scaling factor.
            ctx.scale(b / 50 / 3.5, b / 50 / 3.5);
            ctx.fillStyle = iconColor;
            ctx.fill(buttonCells[j].symbol, 'evenodd');
        }
        ctx.restore();
    }
}

const canvasPadding = 1; // 1px one each side to avoid cutting off outer strokes.

function redraw() {
    const c = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    ctx.resetTransform();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Add 1 pixel breathing space for hex outlines.
    ctx.translate(canvasPadding, canvasPadding);
    if (gstate.board) {
        drawBoard(ctx);
    }
    drawButtons(ctx);
}

function resizeCanvas() {
    const canvas = document.getElementById("canvas");
    const maxWidth = 800;
    // default width and height are assuming a=30 hexz side length.
    const pad = 2 * canvasPadding;
    // 10 hexz of width sqrt(3)*30
    const defaultWidth = 520 + pad;
    // 1px padding t/b + (6*2*30 + 5*30) hexz height + 3*30 for hexz-buttons.
    const defaultHeight = 600 + pad;
    canvas.width = Math.floor(Math.min(
        maxWidth,
        document.body.clientWidth,
        // Ensure the whole board fits on the screen.
        0.9 * document.body.clientHeight / defaultHeight * defaultWidth
    ));
    canvas.height = Math.ceil(canvas.width * (defaultHeight / defaultWidth));
    hexagonSideLength = (canvas.width - pad) / (10 * Math.sqrt(3));
    redraw();
}

function initialize() {
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    const canvas = document.getElementById("canvas");
    canvas.addEventListener('click', onCanvasClicked);
    canvas.addEventListener("contextmenu", e => {
        // Don't open the browser's context menu on right-click.
        e.preventDefault();
        onCanvasClicked(e);
    });
    document.getElementById("home").addEventListener('click', newGame);
    document.getElementById("reset").addEventListener('click', resetGame);
    document.getElementById("undo").addEventListener('click', undoRedoMove("undo"));
    document.getElementById("redo").addEventListener('click', undoRedoMove("redo"));
    document.getElementById("shareLink").addEventListener('click', async function () {
        try {
            await navigator.clipboard.writeText(window.location.href);
            console.log("Copied URL to clipboard.");
        } catch (err) {
            console.log("Cannot copy to clipboard:", err);
        }
    });

    const eventSource = new EventSource(`${URL_PREFIX}/sse/${gameId()}`);
    eventSource.onmessage = (event) => {
        // console.log(`Received event (${event.data.length} bytes)`);
        handleServerEvent(eventSource, JSON.parse(event.data));
    }
    eventSource.onerror = (err) => {
        // Do nothing for now, the Browser will try to reconnect.
    }
}

function onCanvasClicked(event) {
    if (gstate.done) {
        return;  // Do nothing if the game is over.
    }
    if (event.button != 0 || event.altKey || event.metaKey || event.ctrlKey || event.shiftKey) {
        // Just an experiment: how to detect modifier keys and right-clicks.
        const e = event;
        console.log(`Click event: button=${e.button} altKey=${e.altKey} metaKey=${e.metaKey} ctrlKey=${e.ctrlKey} shiftKey=${e.shiftKey}`);
        return;
    }
    // const started = performance.now();
    const c = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    ctx.resetTransform();
    let a = hexagonSideLength;
    let b = Math.sqrt(3) * a;
    let nRows = gstate.board.fields.length;
    let hex = hexagon(a);
    // Detect a click on a cell of the board.
    for (let i = 0; i < nRows; i++) {
        for (let j = 0; j < gstate.board.fields[i].length; j++) {
            let xOff = (i % 2 == 0) ? 0 : b / 2;
            let x = xOff + j * b + b / 2;
            let y = i * a * 3 / 2 + a;
            ctx.translate(x, y);
            if (ctx.isPointInPath(hex, event.offsetX, event.offsetY)) {
                onFieldClicked(i, j);
                break;
            }
            ctx.translate(-x, -y);
        }
    }
    // Detect a click on one of the cell buttons below the board.
    let buttonRow = nRows + 1;
    let xOff = (buttonRow % 2 == 0) ? 0 : b / 2;
    for (let j = 0; j < buttonCells.length; j++) {
        let x = xOff + (j * 1.3) * b + b / 2;
        let y = buttonRow * a * 3 / 2 + a;
        ctx.translate(x, y);
        if (ctx.isPointInPath(hex, event.offsetX, event.offsetY)) {
            onCellButtonClicked(buttonCells[j].type);
            break;
        }
        ctx.translate(-x, -y);
    }
    // console.log("Click processing took ", performance.now() - started, "ms");
}

function onFieldClicked(row, col) {
    if (gstate.board.turn != gstate.role) {
        return; // Do nothing if it's not our turn.
    }
    sendMove(row, col, gstate.selectedCellType);
    // Revert selection to normal to avoid accidentally placing special cells.
    gstate.selectedCellType = cellNormal;
}

function onCellButtonClicked(cellType) {
    let playerIdx = gstate.role - 1;
    let available = cellType == cellNormal
        || gstate.board.resources[playerIdx].numPieces[cellType] != 0;
    if (available) {
        gstate.selectedCellType = cellType;
        redraw();
    }
}

let animationStartedMillis = 0;

function animateWinner() {
    const duration = 5000;
    const waitTime = 3000;
    const freq = duration / 8;
    let elapsed = Date.now() - animationStartedMillis;
    const continueAnimation = elapsed < duration;
    elapsed = Math.min(duration, elapsed);
    const ctx = document.getElementById("canvas").getContext('2d');
    ctx.resetTransform();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const trophy = String.fromCodePoint("0x1F3C6");
    const letters = [trophy, "Y", "O", "U", " ", "W", "O", "N", "!", trophy];
    const fontSize = Math.floor(Math.min(h / 4, w / (letters.length * 1.1)));
    ctx.font = `${fontSize}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = gstate.role > 0 ? styles.colors.players[gstate.role - 1] : "#ffffff";
    for (let i = 0; i < letters.length; i++) {
        const xOff = w / (2 * letters.length) * (2 * i + 1);
        const yOff = h / 2 + h / 4 * Math.sin((elapsed + i * 0.25 * freq) / freq * Math.PI);
        ctx.fillText(letters[i], xOff, yOff);
    }
    if (continueAnimation) {
        window.requestAnimationFrame(animateWinner);
    } else {
        setTimeout(function () { redraw(); }, waitTime);
    }
}

function animateLoser() {
    const duration = 5000;
    const waitTime = 3000;
    const delay = 1000;
    const freq = duration / 2;
    let elapsed = Date.now() - animationStartedMillis;
    const continueAnimation = elapsed < duration;
    elapsed = Math.min(duration, elapsed);
    const delayedX = Math.max(0, (elapsed - delay) / (duration - delay)) * duration;
    const ctx = document.getElementById("canvas").getContext('2d');
    ctx.resetTransform();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const sadFace = String.fromCodePoint("0x1F622");
    const seeNoEvil = String.fromCodePoint("0x1F648");
    const text = sadFace + " YOU LOST " + seeNoEvil;
    const fontSize = Math.floor(Math.min(h / (text.length * 0.8), w / (text.length * 0.8)));
    ctx.font = `${fontSize}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = gstate.role > 0 ? styles.colors.players[gstate.role - 1] : "#ffffff";
    ctx.translate(w / 2, h / 2);
    if (continueAnimation) {
        ctx.rotate(Math.PI * delayedX / freq);
        const scaleFactor = 1 - delayedX / duration;
        ctx.scale(scaleFactor, scaleFactor);
    }
    ctx.fillText(text, 0, 0);
    if (continueAnimation) {
        window.requestAnimationFrame(animateLoser);
    } else {
        setTimeout(function () { redraw(); }, waitTime);
    }
}

function showAnimation(animationFunc) {
    animationStartedMillis = Date.now();
    window.requestAnimationFrame(animationFunc);
}

function reset() {
    redraw();
}

let wasmWorker = null;
function startWASMWebWorker() {
    wasmWorker = new Worker(`${URL_PREFIX}/static/js/wasmworker.js`);
    // First message is initialization (FMII) (since you cannot pass arguments to worker):
    wasmWorker.postMessage({
        urlPrefix: URL_PREFIX,
    })
    // wasmWorker.onmessage gets called when the worker posts a message,
    // which contains the suggested move.
    // This move is sent to the game server. The server will asynchronously
    // send a board update (via SSE), which will show the move in the UI.
    wasmWorker.onmessage = async (e) => {
        move = e.data;
        if (!move) {
            console.log("CPU did not find a move.");
        }
        const moveResponse = await fetch(`${URL_PREFIX}/move/${gameId()}`, {
            method: "POST",
            headers: { "Content-Type": "application/json", },
            body: JSON.stringify(move.moveRequest),
        });
        if (!moveResponse.ok) {
            console.log("Failed to make a move: ", moveResponse.statusText);
            return false;
        }
        // Send WASMStatsRequest so we can learn how our clients perform.
        const statsResponse = await fetch(`${URL_PREFIX}/wasmstats/${gameId()}`, {
            method: "POST",
            headers: { "Content-Type": "application/json", },
            body: JSON.stringify({
                gameId: gameId(),
                gameType: "Flagz",
                move: gstate.board.move,
                userInfo: {
                    userAgent: navigator.userAgent,
                    language: navigator.language,
                    resolution: [window.screen.width, window.screen.height],
                    viewport: [window.innerWidth, window.innerHeight],
                    browserWindow: [window.outerWidth, window.outerHeight],
                    hardwareConcurrency: navigator.hardwareConcurrency,
                },
                stats: move.stats
            }),
        });
        if (!statsResponse.ok) {
            console.log("Failed to send WASM stats: ", statsResponse.statusText);
        }
    }
}

function sendWASMWorkerMoveRequest() {
    wasmWorker.postMessage({
        gameId: gameId(),
    })
}

initialize();
