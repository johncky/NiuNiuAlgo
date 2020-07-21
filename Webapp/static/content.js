var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var Dashboard = function (_React$Component) {
    _inherits(Dashboard, _React$Component);

    function Dashboard(props) {
        _classCallCheck(this, Dashboard);

        var _this = _possibleConstructorReturn(this, (Dashboard.__proto__ || Object.getPrototypeOf(Dashboard)).call(this, props));

        _this.state = { cur_nav: 'content',
            data: props.data,
            cur_data: props.data.algos_data.combined,
            cur_content: 'main'
        };
        _this.change_algo_handler = _this.change_algo.bind(_this);
        _this.change_content_handler = _this.change_content.bind(_this);
        _this.update_data_handelr = _this.update_data.bind(_this);
        _this.update_interval = null;
        return _this;
    }

    _createClass(Dashboard, [{
        key: 'update_data',
        value: function update_data(new_data) {
            var add_algo = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;

            if (add_algo == true) {
                this.setState({ data: new_data, cur_data: new_data.algos_data.combined });
            } else {
                this.setState({ data: new_data });
            }
        }
    }, {
        key: 'componentDidMount',
        value: function componentDidMount() {
            var _this2 = this;

            this.update_interval = setInterval(function () {
                return get_summary(_this2.update_data_handelr);
            }, 1000 * 30);
        }
    }, {
        key: 'componentWillUnmount',
        value: function componentWillUnmount() {
            clearInterval(this.update_interval);
        }
    }, {
        key: 'change_algo',
        value: function change_algo(to_algo) {
            this.setState({ cur_data: this.state.data.algos_data[to_algo] });
        }
    }, {
        key: 'change_content',
        value: function change_content(to_content) {
            if (to_content === 'settings') {
                this.setState({ cur_content: 'settings' });
            } else if (to_content == 'main') {
                this.setState({ cur_content: 'main' });
            } else {
                this.setState({ cur_content: 'datails' });
            }
        }
    }, {
        key: 'render',
        value: function render() {
            var _this3 = this;

            if (this.state.cur_nav === 'content') {
                if (Object.keys(this.state.data.algos_data).length > 0) {

                    var navbar = { brandname: 'Algo.Py', navs: [{ active: true, text: 'Hook', is_dropdown: false, dropdown_content: [], on_click: function on_click() {
                                console.log('aaa');
                            }, text_color: null }, { active: true, text: 'Algo', is_dropdown: true, dropdown_content: Object.keys(this.state.data.algos_data).map(function (x) {
                                return { active: true, text: _this3.state.data.algos_data[x]['name'], is_dropdown: false, dropdown_content: [], on_click: function on_click() {
                                        _this3.change_algo_handler(_this3.state.data.algos_data[x]['name']);
                                    } };
                            }) }] };
                    return React.createElement(
                        'div',
                        null,
                        React.createElement(NavBar, Object.assign({}, navbar, { update_data_handelr: this.update_data_handelr })),
                        React.createElement(Content, { cur_content: this.state.cur_content, algos_data: this.state.cur_data, change_content_handler: this.change_content_handler })
                    );
                } else {
                    var navbar = { brandname: 'Algo.Py', navs: [{ active: true, text: 'Hook', is_dropdown: false, dropdown_content: [], on_click: function on_click() {
                                console.log('aaa');
                            }, text_color: null }] };
                    return React.createElement(
                        'div',
                        null,
                        React.createElement(NavBar, Object.assign({}, navbar, { update_data_handelr: this.update_data_handelr })),
                        React.createElement(NoAlgoPage, null)
                    );
                }
            }
        }
    }]);

    return Dashboard;
}(React.Component);

function NoAlgoPage(props) {
    return React.createElement(
        'h2',
        null,
        'No Running Algo Found!'
    );
}
function SubPageNav(props) {
    return React.createElement(
        'li',
        { className: 'nav-link active text-dark', key: props.to_content },
        React.createElement(
            'a',
            { className: 'nav-link active text-dark btn font-weight-bold', onClick: props.handler },
            props.to_content
        )
    );
}

function SubPageNavBar(props) {
    return React.createElement(
        'div',
        { className: 'container-fluid navbar navbar-expand navbar-dark flex-column flex-md-row bd-navbar bg-light' },
        React.createElement(
            'div',
            { className: 'collapse navbar-collapse', id: 'navbarSupportedContent' },
            React.createElement(
                'a',
                { className: 'navbar-brand text-dark font-weight-bold' },
                props.algo_name
            ),
            React.createElement(
                'ul',
                { className: 'navbar-nav mr-auto' },
                React.createElement(SubPageNav, { to_content: 'Main', handler: function handler() {
                        return props.handler('main');
                    } }),
                React.createElement(SubPageNav, { to_content: 'Settings', handler: function handler() {
                        return props.handler('settings');
                    } }),
                React.createElement(SubPageNav, { to_content: 'Details', handler: function handler() {
                        return props.handler('details');
                    } })
            )
        )
    );
}

var Content = function (_React$Component2) {
    _inherits(Content, _React$Component2);

    function Content(props) {
        _classCallCheck(this, Content);

        return _possibleConstructorReturn(this, (Content.__proto__ || Object.getPrototypeOf(Content)).call(this, props));
    }

    _createClass(Content, [{
        key: 'render',
        value: function render() {
            if (this.props.cur_content === 'settings') {
                return React.createElement(
                    'main',
                    { className: 'main', id: 'main' },
                    React.createElement(
                        'div',
                        { className: 'container-fluid bg-light' },
                        React.createElement(
                            CardDeck,
                            null,
                            React.createElement(
                                Card,
                                { style: { width: '100%' }, key: '1', className: 'pblank-4 rounded shadow', bg: 'light', text: 'dark', border: 'light' },
                                React.createElement(
                                    Card.Body,
                                    null,
                                    React.createElement(SubPageNavBar, { algo_name: this.props.algos_data.name, handler: this.props.change_content_handler }),
                                    React.createElement(Settings, { settings: this.props.algos_data.settings })
                                )
                            )
                        )
                    )
                );
            } else if (this.props.cur_content === "main") {
                return React.createElement(
                    'main',
                    { className: 'main', id: 'main' },
                    React.createElement(
                        'div',
                        { className: 'container-fluid bg-light' },
                        React.createElement(
                            CardDeck,
                            null,
                            React.createElement(
                                Card,
                                { style: { width: '100%' }, key: '1', className: 'pblank-4 rounded shadow', bg: 'light', text: 'dark', border: 'light' },
                                React.createElement(
                                    Card.Body,
                                    null,
                                    React.createElement(SubPageNavBar, { algo_name: this.props.algos_data.name, handler: this.props.change_content_handler }),
                                    React.createElement(DashboardMain, { algos_data: this.props.algos_data })
                                )
                            )
                        )
                    )
                );
            } else if (this.props.cur_content === 'datails') {
                return React.createElement(
                    'main',
                    { className: 'main', id: 'main' },
                    React.createElement(
                        'div',
                        { className: 'container-fluid bg-light' },
                        React.createElement(
                            CardDeck,
                            null,
                            React.createElement(
                                Card,
                                { style: { width: '100%' }, key: '1', className: 'pblank-4 rounded shadow', bg: 'light', text: 'dark', border: 'light' },
                                React.createElement(
                                    Card.Body,
                                    null,
                                    React.createElement(SubPageNavBar, { algo_name: this.props.algos_data.name, handler: this.props.change_content_handler }),
                                    'Details Content'
                                )
                            )
                        )
                    )
                );
            }
        }
    }]);

    return Content;
}(React.Component);

function SettingInputField(props) {

    return React.createElement(
        InputGroup,
        { className: 'mb-2' },
        React.createElement(
            InputGroup.Prepend,
            null,
            React.createElement(
                InputGroup.Text,
                null,
                props.field
            )
        ),
        React.createElement(FormControl, { placeholder: props.value }),
        React.createElement(
            Button,
            { variant: 'outline-secondary', onClick: function onClick() {
                    return console.log('aa');
                } },
            'Update'
        )
    );
}

function Settings(props) {
    var settings = props.settings;
    var headers = Object.keys(settings);
    headers = headers.length > 0 ? headers : [];
    return React.createElement(
        CardDeck,
        null,
        React.createElement(
            Card,
            { style: { width: '100%' }, className: 'p-4 md-4 shadow', bg: 'light', text: 'dark', border: 'light' },
            React.createElement(
                Card.Body,
                null,
                React.createElement(
                    Card.Title,
                    null,
                    'Settings'
                ),
                headers.map(function (field) {
                    return React.createElement(SettingInputField, { field: field, value: settings[field], key: field });
                })
            )
        )
    );
}

var DashboardMain = function (_React$Component3) {
    _inherits(DashboardMain, _React$Component3);

    function DashboardMain(props) {
        _classCallCheck(this, DashboardMain);

        return _possibleConstructorReturn(this, (DashboardMain.__proto__ || Object.getPrototypeOf(DashboardMain)).call(this, props));
    }

    _createClass(DashboardMain, [{
        key: 'random_graph',
        value: function random_graph() {}
    }, {
        key: 'componentDidMount',
        value: function componentDidMount() {
            var _this6 = this;

            this.interval = setInterval(function () {
                return _this6.random_graph.bind(_this6)();
            }, 1000);
        }
    }, {
        key: 'componentWillUnmount',
        value: function componentWillUnmount() {
            clearInterval(this.interval);
        }
    }, {
        key: 'render',
        value: function render() {

            return React.createElement(
                'div',
                null,
                React.createElement(
                    CardDeck,
                    null,
                    React.createElement(
                        Card,
                        { style: { width: '30%' }, className: 'p-4 md-4 shadow', bg: 'light', text: 'dark', border: 'light' },
                        React.createElement(
                            Card.Body,
                            null,
                            React.createElement(
                                Card.Title,
                                null,
                                'Performance'
                            ),
                            React.createElement(PerformanceCard, { algos_data: this.props.algos_data })
                        )
                    ),
                    React.createElement(
                        Card,
                        { style: { width: '30%' }, className: 'p-4 md-4 shadow', bg: 'light', text: 'dark', border: 'light' },
                        React.createElement(
                            Card.Body,
                            null,
                            React.createElement(
                                Card.Title,
                                null,
                                'Status'
                            ),
                            React.createElement(StatsCard, { algos_data: this.props.algos_data })
                        )
                    ),
                    React.createElement(
                        Card,
                        { style: { width: '30%' }, className: 'p-4 md-4 shadow', bg: 'light', text: 'dark', border: 'light' },
                        React.createElement(
                            Card.Body,
                            null,
                            React.createElement(
                                Card.Title,
                                null,
                                'Charts'
                            ),
                            React.createElement(ChartCard, { algos_data: this.props.algos_data })
                        )
                    )
                ),
                React.createElement(
                    CardDeck,
                    null,
                    React.createElement(
                        Card,
                        { style: { width: '100%' }, className: 'pblank-1  mt-3 shadow', bg: 'light', text: 'dark', border: 'light' },
                        React.createElement(
                            Card.Body,
                            null,
                            React.createElement(
                                Card.Title,
                                null,
                                'Positions'
                            ),
                            React.createElement(ScrollableTable, { headers: this.props.algos_data.positions.length > 0 ? Object.keys(this.props.algos_data.positions[0]) : [], data: this.props.algos_data.positions.map(function (x) {
                                    return Object.values(x);
                                }) })
                        )
                    )
                ),
                React.createElement(
                    CardDeck,
                    null,
                    React.createElement(
                        Card,
                        { style: { width: '100%' }, className: 'pblank-1  mt-3 shadow', bg: 'light', text: 'dark', border: 'light' },
                        React.createElement(
                            Card.Body,
                            null,
                            React.createElement(
                                Card.Title,
                                null,
                                'Pending Orders'
                            ),
                            React.createElement(ScrollableTable, { headers: this.props.algos_data.pending.length > 0 ? Object.keys(this.props.algos_data.pending[0]) : [], data: this.props.algos_data.pending.map(function (x) {
                                    return Object.values(x);
                                }) })
                        )
                    )
                ),
                React.createElement(
                    CardDeck,
                    null,
                    React.createElement(
                        Card,
                        { style: { width: '100%' }, className: 'pblank-1  mt-3 shadow', bg: 'light', text: 'dark', border: 'light' },
                        React.createElement(
                            Card.Body,
                            null,
                            React.createElement(
                                Card.Title,
                                null,
                                'Completed Orders'
                            ),
                            React.createElement(ScrollableTable, { headers: this.props.algos_data.completed.length > 0 ? Object.keys(this.props.algos_data.completed[0]) : [], data: this.props.algos_data.completed.map(function (x) {
                                    return Object.values(x);
                                }) })
                        )
                    )
                )
            );
        }
    }]);

    return DashboardMain;
}(React.Component);

function round_pct(pct, decimals) {
    return Math.round(pct * Math.pow(10, decimals)) / Math.pow(10, decimals);
}

function PerformanceCard(props) {
    var caret = function caret(_return) {
        return _return >= 0 ? React.createElement(
            'svg',
            { className: 'bi bi-caret-up-fill', width: '1em', height: '1em', viewBox: '0 2.5 16 16', fill: 'currentColor', xmlns: 'http://www.w3.org/2000/svg' },
            React.createElement('path', { d: 'M7.247 4.86l-4.796 5.481c-.566.647-.106 1.659.753 1.659h9.592a1 1 0 0 0 .753-1.659l-4.796-5.48a1 1 0 0 0-1.506 0z' })
        ) : React.createElement(
            'svg',
            { className: 'bi bi-caret-down-fill', width: '1em', height: '1em', viewBox: '0 2.5 16 16', fill: 'currentColor', xmlns: 'http://www.w3.org/2000/svg' },
            React.createElement('path', { d: 'M7.247 11.14L2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z' })
        );
    };

    var cal_return_data = function cal_return_data(_return, _return_pct, is_benchmark, pv) {
        return is_benchmark ? { return: round_pct(_return_pct * (pv - _return), 2),
            text_color: _return_pct >= 0 ? 'text-success font-weight-bold d-inline' : 'text-danger font-weight-bold d-inline',
            variant: _return_pct >= 0 ? 'success' : 'danger',
            return_str: '$' + Math.abs(round_pct(_return_pct * (pv - _return), 2).toString()) + '   ',
            return_pct_str: round_pct(_return_pct * 100, 2).toString() + '%',
            sign: _return_pct >= 0 ? '+' : '-',
            caret: caret(_return_pct) } : { return: _return,
            text_color: _return_pct >= 0 ? 'text-success font-weight-bold d-inline' : 'text-danger font-weight-bold d-inline',
            variant: _return_pct >= 0 ? 'success' : 'danger',
            return_str: '$' + Math.abs(round_pct(_return, 2).toString()) + '   ',
            return_pct_str: round_pct(_return_pct * 100, 2).toString() + '%',
            sign: _return_pct >= 0 ? '+' : '-',
            caret: caret(_return_pct) };
    };

    var daily = cal_return_data(props.algos_data.daily_return, props.algos_data.daily_return_pct, false, 0);
    var benchmark_daily = cal_return_data(props.algos_data.daily_return, props.algos_data.benchmark_daily_pct, true, props.algos_data.pv);
    var monthly = cal_return_data(props.algos_data.monthly_return, props.algos_data.monthly_return_pct, false, 0);
    var benchmark_monthly = cal_return_data(props.algos_data.monthly_return, props.algos_data.benchmark_monthly_pct, true, props.algos_data.pv);
    var total = cal_return_data(props.algos_data.net_pnl, props.algos_data.net_pnl_pct, false, 0);
    var total_benchmark = cal_return_data(props.algos_data.net_pnl, props.algos_data.benchmark_net_pnl_pct, true, props.algos_data.pv);

    return React.createElement(
        'div',
        null,
        React.createElement('hr', null),
        React.createElement(
            'h6',
            null,
            ' Daily PnL: '
        ),
        React.createElement(
            'h5',
            { className: daily.text_color },
            daily.sign,
            daily.return_str
        ),
        React.createElement(
            'h5',
            { className: 'd-inline mh-100' },
            React.createElement(
                Badge,
                { pill: true, variant: daily.variant },
                daily.caret,
                ' ',
                daily.return_pct_str
            )
        ),
        React.createElement(
            'h3',
            { className: 'd-inline font-weight-light' },
            '   /  '
        ),
        React.createElement(
            'h5',
            { className: benchmark_daily.text_color },
            benchmark_daily.sign,
            benchmark_daily.return_str,
            '(',
            props.algos_data.benchmark,
            ')'
        ),
        React.createElement(
            'h5',
            { className: 'd-inline mh-100' },
            React.createElement(
                Badge,
                { pill: true, variant: benchmark_daily.variant },
                benchmark_daily.caret,
                ' ',
                benchmark_daily.return_pct_str
            )
        ),
        React.createElement('hr', null),
        React.createElement(
            'h6',
            null,
            ' Monthly PnL: '
        ),
        React.createElement(
            'h5',
            { className: monthly.text_color },
            monthly.sign,
            monthly.return_str
        ),
        React.createElement(
            'h5',
            { className: 'd-inline mh-100' },
            React.createElement(
                Badge,
                { pill: true, variant: monthly.variant },
                monthly.caret,
                ' ',
                monthly.return_pct_str
            )
        ),
        React.createElement(
            'h3',
            { className: 'd-inline font-weight-light' },
            '   /  '
        ),
        React.createElement(
            'h5',
            { className: benchmark_monthly.text_color },
            benchmark_monthly.sign,
            benchmark_monthly.return_str,
            '(',
            props.algos_data.benchmark,
            ')'
        ),
        React.createElement(
            'h5',
            { className: 'd-inline mh-100' },
            React.createElement(
                Badge,
                { pill: true, variant: benchmark_monthly.variant },
                benchmark_monthly.caret,
                ' ',
                benchmark_monthly.return_pct_str
            )
        ),
        React.createElement('hr', null),
        React.createElement(
            'h6',
            null,
            ' Total PnL: '
        ),
        React.createElement(
            'h5',
            { className: total.text_color },
            total.sign,
            total.return_str
        ),
        React.createElement(
            'h5',
            { className: 'd-inline mh-100' },
            React.createElement(
                Badge,
                { pill: true, variant: total.variant },
                total.caret,
                ' ',
                total.return_pct_str
            )
        ),
        React.createElement(
            'h3',
            { className: 'd-inline font-weight-light' },
            '   /  '
        ),
        React.createElement(
            'h5',
            { className: total_benchmark.text_color },
            total_benchmark.sign,
            total_benchmark.return_str,
            '(',
            props.algos_data.benchmark,
            ')'
        ),
        React.createElement(
            'h5',
            { className: 'd-inline mh-100' },
            React.createElement(
                Badge,
                { pill: true, variant: total_benchmark.variant },
                total_benchmark.caret,
                ' ',
                total_benchmark.return_pct_str
            )
        ),
        React.createElement('hr', null),
        React.createElement(
            Table,
            { hover: true },
            React.createElement(
                'thead',
                null,
                React.createElement(
                    'tr',
                    null,
                    React.createElement(
                        'th',
                        null,
                        'Range'
                    ),
                    React.createElement(
                        'th',
                        null,
                        'Actual'
                    ),
                    React.createElement(
                        'th',
                        null,
                        'Benchmark'
                    ),
                    React.createElement(
                        'th',
                        null,
                        'Outperformance'
                    )
                )
            ),
            React.createElement(
                'tbody',
                null,
                React.createElement(
                    'tr',
                    null,
                    React.createElement(
                        'td',
                        null,
                        'Annualized Return: '
                    ),
                    React.createElement(
                        'td',
                        null,
                        round_pct(props.algos_data.annualized_return * 100, 2).toString() + '% '
                    ),
                    React.createElement(
                        'td',
                        null,
                        round_pct(props.algos_data.benchmark_annualized_return * 100, 2).toString() + '% '
                    ),
                    React.createElement(
                        'td',
                        null,
                        round_pct((props.algos_data.annualized_return - props.algos_data.benchmark_annualized_return) * 100, 2) + '%'
                    )
                ),
                React.createElement(
                    'tr',
                    null,
                    React.createElement(
                        'td',
                        null,
                        'Daily: '
                    ),
                    React.createElement(
                        'td',
                        null,
                        daily.return_pct_str
                    ),
                    React.createElement(
                        'td',
                        null,
                        benchmark_daily.return_pct_str
                    ),
                    React.createElement(
                        'td',
                        null,
                        round_pct((props.algos_data.daily_return_pct - props.algos_data.benchmark_daily_pct) * 100, 2) + '%'
                    )
                ),
                React.createElement(
                    'tr',
                    null,
                    React.createElement(
                        'td',
                        null,
                        'Monthly: '
                    ),
                    React.createElement(
                        'td',
                        null,
                        monthly.return_pct_str
                    ),
                    React.createElement(
                        'td',
                        null,
                        benchmark_monthly.return_pct_str
                    ),
                    React.createElement(
                        'td',
                        null,
                        round_pct((props.algos_data.monthly_return_pct - props.algos_data.benchmark_monthly_pct) * 100, 2) + '%'
                    )
                )
            )
        )
    );
};

function StatsCard(props) {
    var button = props.algos_data.status.toLowerCase() == 'running' ? React.createElement(
        'svg',
        { className: 'bi bi-x-circle', width: '1em', height: '1em', viewBox: '0 0 16 16', fill: 'currentColor', xmlns: 'http://www.w3.org/2000/svg' },
        React.createElement('path', { fillRule: 'evenodd', d: 'M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z' }),
        React.createElement('path', { fillRule: 'evenodd', d: 'M11.854 4.146a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708-.708l7-7a.5.5 0 0 1 .708 0z' }),
        React.createElement('path', { fillRule: 'evenodd', d: 'M4.146 4.146a.5.5 0 0 0 0 .708l7 7a.5.5 0 0 0 .708-.708l-7-7a.5.5 0 0 0-.708 0z' })
    ) : React.createElement(
        'svg',
        { className: 'bi bi-play', width: '1em', height: '1em', viewBox: '0 0 16 16', fill: 'currentColor', xmlns: 'http://www.w3.org/2000/svg' },
        React.createElement('path', { fillRule: 'evenodd', d: 'M10.804 8L5 4.633v6.734L10.804 8zm.792-.696a.802.802 0 0 1 0 1.392l-6.363 3.692C4.713 12.69 4 12.345 4 11.692V4.308c0-.653.713-.998 1.233-.696l6.363 3.692z' })
    );
    return React.createElement(
        'div',
        null,
        React.createElement('hr', null),
        React.createElement(
            'h6',
            { className: 'd-inline' },
            'Algo Status:  '
        ),
        React.createElement(
            'h6',
            { className: 'd-inline' },
            props.algos_data.status + '   ',
            React.createElement(
                'a',
                { className: 'navbar-brand mr-0 mr-md-0 text-dark', href: '/' },
                button
            )
        ),
        React.createElement('hr', null),
        React.createElement(
            Table,
            { hover: true },
            React.createElement(
                'thead',
                null,
                React.createElement(
                    'tr',
                    null,
                    React.createElement(
                        'th',
                        null,
                        'Portfolio Value:'
                    ),
                    React.createElement(
                        'th',
                        null,
                        '$' + props.algos_data.pv
                    ),
                    React.createElement(
                        'th',
                        null,
                        '%'
                    )
                )
            ),
            React.createElement(
                'tbody',
                null,
                React.createElement(
                    'tr',
                    null,
                    React.createElement(
                        'td',
                        null,
                        'Assets: '
                    ),
                    React.createElement(
                        'td',
                        null,
                        '$' + (props.algos_data.pv - props.algos_data.cash)
                    ),
                    React.createElement(
                        'td',
                        null,
                        round_pct((props.algos_data.pv - props.algos_data.cash) / props.algos_data.pv * 100, 2) + '%'
                    )
                ),
                React.createElement(
                    'tr',
                    null,
                    React.createElement(
                        'td',
                        null,
                        'Cash: '
                    ),
                    React.createElement(
                        'td',
                        null,
                        '$' + props.algos_data.cash
                    ),
                    React.createElement(
                        'td',
                        null,
                        round_pct(props.algos_data.cash / props.algos_data.pv * 100, 2) + '%'
                    )
                )
            )
        ),
        React.createElement('hr', null),
        React.createElement(
            Table,
            { hover: true },
            React.createElement(
                'thead',
                null,
                React.createElement(
                    'tr',
                    null,
                    React.createElement(
                        'th',
                        null,
                        'Metrics \xA0 \xA0  \xA0 \xA0 \xA0 '
                    ),
                    React.createElement(
                        'th',
                        null,
                        props.algos_data.name,
                        '\xA0\xA0  '
                    ),
                    React.createElement(
                        'th',
                        null,
                        props.algos_data.benchmark
                    )
                )
            ),
            React.createElement(
                'tbody',
                null,
                React.createElement(
                    'tr',
                    null,
                    React.createElement(
                        'td',
                        null,
                        'Sharpe:'
                    ),
                    React.createElement(
                        'td',
                        null,
                        round_pct(props.algos_data.sharpe, 2)
                    ),
                    React.createElement(
                        'td',
                        null,
                        round_pct(props.algos_data.benchmark_sharpe, 2)
                    )
                ),
                React.createElement(
                    'tr',
                    null,
                    React.createElement(
                        'td',
                        null,
                        'Beta: '
                    ),
                    React.createElement(
                        'td',
                        null,
                        round_pct(props.algos_data.beta, 2)
                    ),
                    React.createElement(
                        'td',
                        null,
                        1
                    )
                ),
                React.createElement(
                    'tr',
                    null,
                    React.createElement(
                        'td',
                        null,
                        'Sortino: '
                    ),
                    React.createElement(
                        'td',
                        null,
                        'In progress'
                    ),
                    React.createElement(
                        'td',
                        null,
                        'In Progress'
                    )
                ),
                React.createElement(
                    'tr',
                    null,
                    React.createElement(
                        'td',
                        null,
                        'Txn Cost:  '
                    ),
                    React.createElement(
                        'td',
                        null,
                        props.algos_data.txn_cost_total
                    )
                )
            )
        )
    );
}

function ChartCard(props) {
    return React.createElement(
        'div',
        null,
        React.createElement(
            Tabs,
            { defaultActiveKey: 'PV', id: 'uncontrolled-tab-example', className: 'myClass' },
            React.createElement(
                Tab,
                { eventKey: 'PV', title: 'PV' },
                React.createElement(SplineAreaChart, { title: 'Portfolio Value', data: props.algos_data.PV })
            ),
            React.createElement(
                Tab,
                { eventKey: 'EV', title: 'EV' },
                React.createElement(SplineAreaChart, { title: 'Equity Value', data: props.algos_data.EV })
            ),
            React.createElement(
                Tab,
                { eventKey: 'Cash', title: 'Cash' },
                React.createElement(SplineAreaChart, { title: 'Cash', data: props.algos_data.Cash })
            )
        )
    );
}

function Postions(props) {
    return React.createElement('div', null);
}

ReactDOM.render(React.createElement(Dashboard, { data: data }), document.getElementById("main_holder"));