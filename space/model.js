// The trained model, re-run in the browser for the what-if sliders.
// Same maths as space/export.py `replay`, which checks it against XGBoost before export.
(function (root) {
  function vector(model, x) {
    return model.columns.map(function (c) {
      if (c.indexOf("port_") === 0) return x.port === c.slice(5) ? 1 : 0;
      return x[c] === null || x[c] === undefined ? NaN : x[c];
    });
  }

  function treeValue(t, v) {
    var k = 0;
    while (t.f[k] >= 0) {
      var a = v[t.f[k]];
      // XGBoost compares in 32-bit floats; missing values follow the "missing" branch
      k = Number.isNaN(a) ? t.m[k] : (Math.fround(a) < Math.fround(t.t[k]) ? t.y[k] : t.n[k]);
    }
    return t.v[k];
  }

  function score(model, x) {
    var v = vector(model, x);
    var probs = model.members.map(function (m) {
      var margin = m.base_margin;
      for (var i = 0; i < m.trees.length; i++) margin += treeValue(m.trees[i], v);
      return 1 / (1 + Math.exp(-margin));
    });
    var mean = probs.reduce(function (a, b) { return a + b; }, 0) / probs.length;
    var spread = Math.sqrt(probs.reduce(function (a, p) { return a + (p - mean) * (p - mean); }, 0) / probs.length);
    var avg = Math.min(Math.max(mean, 1e-4), 1 - 1e-4);
    var z = model.cal_coef * Math.log(avg / (1 - avg)) + model.cal_intercept;
    var conf = spread < model.cuts[0] ? "High" : spread < model.cuts[1] ? "Medium" : "Low";
    return { prob: 1 / (1 + Math.exp(-z)), spread: spread, conf: conf };
  }

  function backlogDays(waiting, berths7d) {
    return waiting / Math.max(berths7d / 7, 0.5);  // Little's law, as in src/features.py
  }

  root.PortModel = { score: score, backlogDays: backlogDays };
})(typeof window !== "undefined" ? window : globalThis);
