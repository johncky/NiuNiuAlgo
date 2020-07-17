var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var SplineAreaChart = function (_React$Component) {
    _inherits(SplineAreaChart, _React$Component);

    function SplineAreaChart(props) {
        _classCallCheck(this, SplineAreaChart);

        return _possibleConstructorReturn(this, (SplineAreaChart.__proto__ || Object.getPrototypeOf(SplineAreaChart)).call(this, props));
    }

    _createClass(SplineAreaChart, [{
        key: "render",
        value: function render() {

            Xformat = this.props.Xlegend ? this.props.Xformat : "MM DD";
            unit = this.props.unit ? this.props.unit : "$";
            Xlegend = this.props.Xlegend ? this.props.Xlegend : "Datetime";
            Ylegend = this.props.Ylegend ? this.props.Ylegend : unit;

            var options = {
                backgroundColor: "#f8f9fa",
                theme: 'light',
                animationEnabled: true,
                zoomEnabled: true,
                title: {
                    text: this.props.title
                },
                axisY: {
                    gridThickness: 0,
                    tickLength: 0,
                    lineThickness: 0,
                    title: Ylegend,
                    includeZero: false,
                    suffix: ""
                },
                data: [{
                    color: 'dark',
                    lineThickness: 0,
                    fillOpacity: 0.7,
                    markerSize: 0,
                    type: "area",
                    xValueFormatString: Xformat,
                    yValueFormatString: "#,##0.##",
                    showInLegend: true,
                    legendText: Xlegend,
                    dataPoints: this.props.data.map(function (item) {
                        return { x: new Date(item.x), y: Number(item.y) };
                    })
                }]
            };
            return React.createElement(
                "div",
                { className: "mt-5 mx-0" },
                React.createElement(CanvasJSChart, { options: options })
            );
        }
    }]);

    return SplineAreaChart;
}(React.Component);

function ScrollableTable(props) {
    return React.createElement(
        "div",
        { className: "scrollable_table" },
        React.createElement(
            Table,
            { striped: true, bordered: true, hover: true, size: "sm" },
            React.createElement(
                "thead",
                null,
                React.createElement(
                    "tr",
                    null,
                    props.headers.map(function (h) {
                        return React.createElement(
                            "th",
                            { key: h },
                            h
                        );
                    })
                )
            ),
            React.createElement(
                "tbody",
                null,
                props.data.map(function (h, row_id) {
                    return React.createElement(
                        "tr",
                        { key: row_id },
                        h.map(function (x, col_id) {
                            return React.createElement(
                                "td",
                                { key: row_id.toString() + col_id.toString() },
                                x
                            );
                        })
                    );
                })
            )
        )
    );
}